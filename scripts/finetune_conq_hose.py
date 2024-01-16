#!/usr/bin/env python3
from pathlib import Path

import flax
import jax
import optax
import tensorflow as tf
import tqdm
import wandb
from absl import app, flags, logging

from octo.data.dataset import make_single_dataset
from octo.data.utils.data_utils import NormalizationType
from octo.model.components.action_heads import L1ActionHead
from octo.model.components.tokenizers import LowdimObsTokenizer
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import (
    freeze_weights,
    merge_params,
    process_text,
    TrainState,
)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "resume_from", None, "Path to an existing fine-tuning checkpoint directory."
)
flags.DEFINE_string("data_dir", None, "Path to finetuning dataset, in RLDS format.")
flags.DEFINE_string("save_dir", None, "Directory for saving finetuning checkpoints.")
flags.DEFINE_integer("batch_size", 32, "Batch size for finetuning.")
flags.DEFINE_integer("n_steps", 200_000, "Number of batches to finetune for.")
flags.DEFINE_integer("pred_horizon", 50, "Number of batches to finetune for.")
flags.DEFINE_integer("obs_window_size", 5, "Number of input observations to condition on.")

flags.DEFINE_bool(
    "freeze_transformer",
    False,
    "Whether pre-trained transformer weights should be frozen.",
)


def main(_):
    assert (FLAGS.batch_size % jax.device_count() == 0), "Batch size must be divisible by device count."

    use_proprio = False

    dataset_name = "conq_hose_manipulation:1.5.0"

    initialize_compilation_cache()
    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    # setup wandb for logging
    wandb.init(project="octo")
    wandb.config.update(flags.FLAGS)
    wandb.config["dataset_name"] = dataset_name

    # load pre-trained model
    if FLAGS.resume_from is None:
        logging.info("Loading base model from HuggingFace...")
        base_model = "hf://rail-berkeley/octo-base"
        pretrained_model = OctoModel.load_pretrained(base_model)
    else:
        pretrained_model = OctoModel.load_pretrained(FLAGS.resume_from)

    if FLAGS.data_dir is None:
        data_dir = Path("~/tensorflow_datasets").expanduser()
    else:
        data_dir = FLAGS.data_dir

    # make finetuning dataset
    # apply Gaussian normalization, load chunks of `pred_horizon` actions since we'll train with action chunking
    # delete goal images in the data loader since we will train a language-conditioned-only policy
    logging.info("Loading finetuning dataset...")
    dataset_kwargs = {
        'name': dataset_name,
        'data_dir': data_dir,
        # QUESTION: how do these keys relate to the dataset or the model head names?
        'image_obs_keys': {"wrist": "hand_color_image", "primary": "frontright_fisheye_image"},
        'state_obs_keys': ["state"] if use_proprio else None,  # I think this key needs to match the dataset
        'language_key': "language_instruction",
        'action_proprio_normalization_type': NormalizationType.NORMAL,
        'absolute_action_mask': [False, False, False, False, False, False, True, True],
    }
    dataset = make_single_dataset(
        dataset_kwargs=dataset_kwargs,
        traj_transform_kwargs=dict(
            window_size=FLAGS.obs_window_size,
            future_action_window_size=FLAGS.pred_horizon - 1,  # so we get pred_horizon actions for our action chunk
        ),
        frame_transform_kwargs=dict(
            resize_size={
                "primary": (256, 256),
                "wrist": (256, 256),
            },
        ),
        train=True,
    )
    train_data_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(1000)
        .batch(FLAGS.batch_size)
        .iterator()
    )

    # run text tokenizer over batch (this needs to happen before training / sharding) + delete unused keys
    text_processor = pretrained_model.text_processor

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    train_data_iter = map(process_batch, train_data_iter)

    if FLAGS.resume_from is None:
        model = modify_model(pretrained_model, dataset, train_data_iter, use_proprio)
        del pretrained_model
    else:
        model = pretrained_model

    model.config['dataset_kwargs'].update(dataset_kwargs)
    wandb.config["dataset_kwargs"] = dataset_kwargs
    fine_tune(model, train_data_iter)


def modify_model(pretrained_model, dataset, train_data_iter, use_proprio):
    example_batch = next(train_data_iter)

    # start from pre-training config and modify
    config = pretrained_model.config
    # del config["model"]["observation_tokenizers"]["wrist"]
    if use_proprio:
        config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
            LowdimObsTokenizer,
            n_bins=256,
            bin_type="normal",
            # FIXME: check what low and high should be!
            low=-2.0,
            high=2.0,
            # FIXME: why is this "proprio" and not "state"? is it to match the named used in the dataset?
            obs_keys=["proprio"],
        )
    # Fully override the old action head with a new one (for smaller changes, you can use update_module_config)
    config["model"]["heads"]["action"] = ModuleSpec.create(
        L1ActionHead,
        pred_horizon=FLAGS.pred_horizon,
        action_dim=example_batch["action"].shape[-1],
        readout_key="readout_action",
    )
    # initialize weights for modified Octo model, then merge in all applicable pre-trained weights
    # new position encodings for proprio inputs & weights for new action head will remain "from scratch"
    logging.info("Updating model for new observation & action space...")
    model = OctoModel.from_config(
        config,
        example_batch,
        pretrained_model.text_processor,
        verbose=True,
        dataset_statistics=dataset.dataset_statistics,
    )
    merged_params = merge_params(model.params, pretrained_model.params)
    # can perform any additional parameter surgery here...
    # ...
    model = model.replace(params=merged_params)
    return model


def fine_tune(model, train_data_iter):
    # create optimizer & train_state, optionally freeze keys for pre-trained transformer
    # train_state bundles parameters & optimizers
    learning_rate = optax.join_schedules(
        [optax.linear_schedule(0, 3e-5, 100), optax.constant_schedule(3e-5)], [100]
    )
    tx = optax.adamw(learning_rate)
    frozen_keys = model.config["optimizer"]["frozen_keys"]
    if FLAGS.freeze_transformer:
        frozen_keys.append("BlockTransformer_0")
    tx = freeze_weights(tx, model.params, frozen_keys)
    train_state = TrainState.create(
        rng=jax.random.PRNGKey(1234),
        model=model,
        tx=tx,
    )

    # define loss function and train step
    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,  # Action head knows to pull out the action readout_key
            batch["action"],
            pad_mask=batch["observation"]["pad_mask"],
            train=train,
        )
        return action_loss, action_metrics

    @jax.jit
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    # run finetuning loop
    checkpoint_path = (Path(FLAGS.save_dir) / wandb.run.name).absolute()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Starting finetuning for {FLAGS.n_steps} steps...")
    i = 0
    for i in tqdm.tqdm(range(FLAGS.n_steps), total=FLAGS.n_steps, dynamic_ncols=True):
        batch = next(train_data_iter)
        train_state, update_info = train_step(train_state, batch)
        if (i + 1) % 100 == 0:
            update_info = jax.device_get(update_info)
            wandb.log(
                flax.traverse_util.flatten_dict({"training": update_info}, sep="/"),
                step=i,
            )
            # TODO: log model artifacts to wandb
        if i in [10, 25_000, 50_000]:
            # save checkpoint
            train_state.model.save_pretrained(step=i, checkpoint_path=checkpoint_path)
    # make sure to save final checkpoint
    train_state.model.save_pretrained(step=i, checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    app.run(main)
