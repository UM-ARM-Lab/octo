#!/usr/bin/env python3
import argparse
import pickle
import time
from pathlib import Path
from typing import Optional, Tuple, Dict

import bosdyn.client
import gym
import jax
import numpy as np
import rerun as rr
from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME, HAND_FRAME_NAME
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.lease import LeaseKeepAlive, LeaseClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.math_helpers import SE3Pose, Quat
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from gym.core import ObsType, ActType
from scipy.spatial.transform import Rotation as Rot

from conq.cameras_utils import source_to_fmt, image_to_opencv
from conq.clients import Clients
from conq.data_recorder import get_state_vec
from conq.hand_motion import hand_pose_cmd
from conq.manipulation import open_gripper, blocking_arm_command
from conq.rerun_utils import viz_common_frames, rr_tform
from conq.utils import setup
from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import add_octo_env_wrappers
from vr.constants import ARM_POSE_CMD_DURATION, ARM_POSE_CMD_PERIOD

np.set_printoptions(suppress=True, precision=4)


def my_euler_to_quat(euler):
    q_xyzw = Rot.from_euler("xyz", euler, degrees=False).as_quat().tolist()
    q_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
    return q_wxyz


class ConqGymEnv(gym.Env):

    def __init__(self, clients: Clients, model_dataset_kwargs: Dict):
        self.clients = clients
        self.dataset_kwargs = model_dataset_kwargs
        self.image_obs_keys = self.dataset_kwargs['image_obs_keys']
        # FIXME: why do I have to add this? The model.observation_tokenizers.obs_stack_keys don't match the keys
        self.image_obs_keys = {"image_" + k: v for k, v in self.image_obs_keys.items()}
        self.use_proprio = self.dataset_kwargs['state_obs_keys'] is not None

        obs_space_dict = {}
        self.img_srcs = []
        self.img_fmts = []
        for image_obs_key, img_src in self.image_obs_keys.items():
            if img_src is not None:
                # NOTE: the 100x100 shape is arbritary and will be change by the ResizeImageWrapper
                obs_space_dict[image_obs_key] = gym.spaces.Box(low=0, high=255, shape=(100, 100, 3), dtype=np.uint8)

                self.img_srcs.append(img_src)
                self.img_fmts.append(source_to_fmt(img_src))

        if self.use_proprio:
            obs_space_dict["proprio"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(66,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict(obs_space_dict)
        # FIXME: does this even matter? are high/low/shape used anywhere?
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        obs = self.get_obs()
        return obs, {}

    def get_obs(self):
        t1 = time.time()
        reqs = []
        for src, fmt in zip(self.img_srcs, self.img_fmts):
            req = build_image_request(src, pixel_format=fmt)
            reqs.append(req)
        ress = self.clients.image.get_image(reqs)
        imgs = [image_to_opencv(res) for res in ress]

        obs = {}
        for image_obs_key, img_np in zip(self.image_obs_keys, imgs):
            obs[image_obs_key] = img_np

        if self.use_proprio:
            state = self.clients.state.get_robot_state()
            prioprio = get_state_vec(state)
            obs["proprio"] = prioprio

        t2 = time.time()
        # print(f"getting obs {t2 - t1:.3f}")
        return obs

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        t0 = time.time()

        open_fraction = np.clip(action[6], 0., 1.)

        target_hand_in_vision = self.get_new_hand_in_vision(action)

        # FIXME: VR_CMD_PERIOD is constant, but different models may have been trained with different values
        arm_cmd = RobotCommandBuilder.arm_pose_command_from_pose(target_hand_in_vision.to_proto(),
                                                                 VISION_FRAME_NAME,
                                                                 seconds=ARM_POSE_CMD_DURATION)
        cmd = arm_cmd
        # FIXME: add this back?
        # cmd = add_follow_with_body(arm_cmd)

        # Add gripper command
        gripper_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(open_fraction)
        cmd_with_gripper = RobotCommandBuilder.build_synchro_command(cmd, gripper_cmd)
        # Execute!
        self.clients.command.robot_command(cmd_with_gripper)

        obs = self.get_obs()

        # Sleep to achieve the desired control frequency
        # also nice to visualize the arm moving
        # FIXME: this isn't going to be robust to change in exec_window,
        #  and that 0.01 constant is sort of made up, to approximate the speed of the model...
        sleep_dt = ARM_POSE_CMD_PERIOD - (time.time() - t0) - 0.06
        if sleep_dt > 0:
            time.sleep(sleep_dt)

        dt = time.time() - t0
        rr.log('inner_control_hz', rr.TimeSeriesScalar(1 / dt))

        return obs, 0.0, False, False, {}

    def get_new_hand_in_vision(self, action):
        state = self.clients.state.get_robot_state()
        snapshot = state.kinematic_state.transforms_snapshot
        hand_in_vision = get_a_tform_b(snapshot, VISION_FRAME_NAME, HAND_FRAME_NAME)

        # Action is an 8 dim vector, and if you look at the features.json you'll find its meaning:
        # [ dx, dy, dz, droll, dpitch, dyaw, open_fraction, is_terminal ] all in hand frame
        delta_hand_in_hand = SE3Pose(x=action[0], y=action[1], z=action[2],
                                     rot=Quat(*my_euler_to_quat([action[3], action[4], action[5]])))
        target_hand_in_vision = hand_in_vision * delta_hand_in_hand

        # for viz in rerun
        viz_common_frames(snapshot)
        rr_tform('target_hand', target_hand_in_vision)

        return target_hand_in_vision


def setup_robot_and_model(args, **wrapper_kwargs):
    # load models
    print("Loading model...")
    checkpoint_weights_path = str(args.checkpoint_path.absolute().parent)
    checkpoint_step = int(args.checkpoint_path.name)
    model = OctoModel.load_pretrained(checkpoint_weights_path, checkpoint_step)
    # setup Conq
    sdk = bosdyn.client.create_standard_sdk('EvalClient')
    # robot = sdk.create_robot("192.168.80.3")
    robot = sdk.create_robot("10.0.0.3")
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    rr.init('eval')
    rr.connect()
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    lease_client.take()
    setup(robot)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    clients = Clients(lease=lease_client, state=robot_state_client, manipulation=manipulation_api_client,
                      image=image_client, raycast=None, command=command_client, robot=robot, recorder=None)
    # wrap the robot environment
    env = ConqGymEnv(clients, model.config['dataset_kwargs'])

    from octo.data.utils.data_utils import NormalizationType
    dataset_kwargs = {
        'name': 'conq_hose_manipulation',
        'data_dir': Path("~/tensorflow_datasets").expanduser(),
        # QUESTION: how do these keys relate to the dataset or the model head names?
        'image_obs_keys': {"wrist": "hand_color_image", "primary": "frontright_fisheye_image"},
        'state_obs_keys': None,
        'language_key': "language_instruction",
        'action_proprio_normalization_type': NormalizationType.NORMAL,
        'absolute_action_mask': [False, False, False, False, False, False, True],
    }
    from octo.data.dataset import make_single_dataset
    ft_dataset, full_dataset_name = make_single_dataset(
        dataset_kwargs=dataset_kwargs,
        traj_transform_kwargs=dict(
            window_size=2,
            future_action_window_size=4 - 1,  # so we get pred_horizon actions for our action chunk
        ),
        frame_transform_kwargs=dict(
            resize_size={
                "primary": (256, 256),
                "wrist": (128, 128),
            },
        ),
        train=True,
    )
    ft_dataset.dataset_statistics

    env = add_octo_env_wrappers(env=env,
                                config=model.config,
                                dataset_statistics=ft_dataset.dataset_statistics,
                                normalization_type="normal",
                                exec_horizon=args.exec_horizon,
                                **wrapper_kwargs)
    return clients, env, model


def run_dataset_replay(args):
    clients, env, _ = setup_robot_and_model(args, no_normalization=True)

    pkls_root = Path("../conq_hose_manipulation_dataset_builder/conq_hose_manipulation/pkls")
    with (LeaseKeepAlive(clients.lease, must_acquire=True, return_at_exit=True)):

        for pkl_path in pkls_root.glob("*train*.pkl"):
            with pkl_path.open('rb') as f:
                data = pickle.load(f)

            open_gripper(clients)
            look_cmd = hand_pose_cmd(clients, 0.8, 0, 0.2, 0, np.deg2rad(0), 0, duration=0.5)
            blocking_arm_command(clients, look_cmd)

            obs, _ = env.reset()

            last_t = time.time()
            for step in data['steps']:
                action = step['action']

                env.step(action)
                now = time.time()
                dt = now - last_t
                print(f"dt {dt:.3f}")
                last_t = now
        print("done!")


def run_eval_model(args):
    clients, env, model = setup_robot_and_model(args)

    goal_instruction = "grasp hose"
    rng = jax.random.PRNGKey(1)
    n_episodes = 10
    with (LeaseKeepAlive(clients.lease, must_acquire=True, return_at_exit=True)):

        for episode_idx in range(n_episodes):
            task = model.create_tasks(texts=[goal_instruction])

            # NOTE: should probably match the starting pose of the robot to the starting pose of the dataset
            blocking_stand(clients.command, timeout_sec=10)
            open_gripper(clients)
            look_cmd = hand_pose_cmd(clients, 0.8, 0, 0.2, 0, np.deg2rad(0), 0, duration=0.5)
            blocking_arm_command(clients, look_cmd)

            # reset env
            obs, _ = env.reset()

            # do rollout
            t = 0
            last_t = time.time()
            while t < args.num_timesteps:
                for img_key, img_src in env.image_obs_keys.items():
                    if img_src is not None:
                        rr.log(f"{img_key}", rr.Image(obs[img_key][-1]))

                # get action
                actions = model.sample_actions(jax.tree_map(lambda x: x[None], obs), task, rng=rng)
                actions = actions[0]

                # perform environment step
                # this will block to achieve the right control frequency to match training time
                obs, _, _, truncated, _ = env.step(actions)

                t += 1

                now = time.time()
                dt = now - last_t
                rr.log('outer_control_hz', rr.TimeSeriesScalar(1 / dt))
                # print(f"step dt {dt:.3f}")
                last_t = now

                if truncated:
                    break


def main():
    np.set_printoptions(suppress=True, linewidth=220, precision=4)
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=Path, help="path up to and including the step number")
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--exec-horizon", type=int, default=1)
    args = parser.parse_args()

    # run_dataset_replay(args)
    run_eval_model(args)


if __name__ == "__main__":
    main()
