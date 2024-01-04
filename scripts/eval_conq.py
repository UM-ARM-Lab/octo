from datetime import datetime
import bosdyn.client
from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME
from bosdyn.client.lease import LeaseKeepAlive, LeaseClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.robot_state import RobotStateClient
from conq.clients import Clients
from functools import partial
import os
import time
from typing import Optional, Tuple

from absl import app, flags, logging
import click
import cv2
import imageio
import jax
import jax.numpy as jnp
import numpy as np
from bosdyn.client.image import ImageClient
from conq.hand_motion import hand_pose_cmd_in_frame
from conq.manipulation import open_gripper, add_follow_with_body
from conq.utils import setup_and_stand
from gym.core import ObsType, ActType

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, RHCWrapper, UnnormalizeActionProprio

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "checkpoint_weights_path", None, "Path to checkpoint", required=True
)
flags.DEFINE_integer("checkpoint_step", None, "Checkpoint step", required=True)

# custom to bridge_data_robot
flags.DEFINE_string("ip", "localhost", "IP address of the robot")
flags.DEFINE_integer("port", 5556, "Port of the robot")
flags.DEFINE_spaceseplist("goal_eep", [0.3, 0.0, 0.15], "Goal position")
flags.DEFINE_spaceseplist("initial_eep", [0.3, 0.0, 0.15], "Initial position")
flags.DEFINE_bool("blocking", False, "Use the blocking controller")

flags.DEFINE_integer("im_size", None, "Image size", required=True)
flags.DEFINE_string("video_save_path", None, "Path to save video")
flags.DEFINE_integer("num_timesteps", 120, "num timesteps")
flags.DEFINE_integer("horizon", 1, "Observation history length")
flags.DEFINE_integer("pred_horizon", 1, "Length of action sequence from model")
flags.DEFINE_integer("exec_horizon", 1, "Length of action sequence to execute")

# show image flag
flags.DEFINE_bool("show_image", False, "Show image")

##############################################################################

STEP_DURATION_MESSAGE = """
Bridge data was collected with non-blocking control and a step duration of 0.2s.
However, we relabel the actions to make it look like the data was collected with
blocking control and we evaluate with blocking control.
We also use a step duration of 0.4s to reduce the jerkiness of the policy.
Be sure to change the step duration back to 0.2 if evaluating with non-blocking control.
"""
STEP_DURATION = 0.4
STICKY_GRIPPER_NUM_STEPS = 1
WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]
CAMERA_TOPICS = [{"name": "/blue/image_raw"}]
ENV_PARAMS = {
    "camera_topics": CAMERA_TOPICS,
    "override_workspace_boundaries": WORKSPACE_BOUNDS,
    "move_duration": STEP_DURATION,
}

##############################################################################


import gym


class ConqGymEnv(gym.Env):
    
    def __init__(self, im_size, blocking):
        self.im_size = im_size
        self.blocking = blocking
        
        # TODO: setup conq

    def robot_code(self):
        sdk = bosdyn.client.create_standard_sdk('EvalClient')
        robot = sdk.create_robot("182.168.80.3")
        bosdyn.client.util.authenticate(robot)
        robot.time_sync.wait_for_sync()

        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
        image_client = robot.ensure_client(ImageClient.default_service_name)

        lease_client.take()

        with (LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)):
            command_client = setup_and_stand(robot)
            clients = Clients(lease=lease_client, state=robot_state_client, manipulation=manipulation_api_client,
                              image=image_client, raycast=None, command=command_client, robot=robot, recorder=None)

            cmd = RobotCommandBuilder.arm_ready_command()
            clients.command.robot_command(cmd)
            open_gripper(clients)

            transforms0 = clients.state.get_robot_state().kinematic_state.transforms_snapshot
            body_in_vision0 = get_a_tform_b(transforms0, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
            
            x, y, z, roll, pitch, yaw = pose

            arm_cmd = hand_pose_cmd_in_frame(body_in_vision0, x, y, z, roll, pitch, yaw, duration=2 * STEP_DURATION)
            cmd = add_follow_with_body(arm_cmd)
            cmd.synchronized_command.arm_command.arm_cartesian_command.max_linear_velocity.value = 0.3
            clients.command.robot_command(cmd)

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        obs = {
            'image_primary': np.zeros((self.im_size, self.im_size, 3), dtype=np.uint8),
        }
        return obs, {}

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        obs = {
            'image_primary': np.zeros((self.im_size, self.im_size, 3), dtype=np.uint8),
        }
        trunc = False
        return obs, 0.0, False, trunc, {}


def main(_):
    # load models
    model = OctoModel.load_pretrained(
        FLAGS.checkpoint_weights_path,
        FLAGS.checkpoint_step,
    )

    env = ConqGymEnv(FLAGS.im_size, FLAGS.blocking)

    # wrap the robot environment
    env = UnnormalizeActionProprio(
        env, model.dataset_statistics, normalization_type="normal"
    )
    env = HistoryWrapper(env, FLAGS.horizon)
    env = RHCWrapper(env, FLAGS.exec_horizon)

    # create policy functions
    @jax.jit
    def sample_actions(
            pretrained_model: OctoModel,
            observations,
            tasks,
            rng,
    ):
        # add batch dim to observations
        observations = jax.tree_map(lambda x: x[None], observations)
        actions = pretrained_model.sample_actions(
            observations,
            tasks,
            rng=rng,
        )
        # remove batch dim
        return actions[0]

    policy_fn = partial(
        sample_actions,
        model,
        argmax=FLAGS.deterministic,
        temperature=FLAGS.temperature,
    )

    goal_image = jnp.zeros((FLAGS.im_size, FLAGS.im_size, 3), dtype=np.uint8)
    goal_instruction = ""

    # goal sampling loop
    while True:
        print("Current instruction: ", goal_instruction)
        if click.confirm("Take a new instruction?", default=True):
            text = input("Instruction?")
        # Format task for the model
        task = model.create_tasks(texts=[text])
        # For logging purposes
        goal_instruction = text
        goal_image = jnp.zeros_like(goal_image)

        # reset env
        obs, _ = env.reset()
        time.sleep(2.0)

        input("Press [Enter] to start.")

        # do rollout
        last_tstep = time.time()
        images = []
        goals = []
        t = 0
        while t < FLAGS.num_timesteps:
            if time.time() > last_tstep + STEP_DURATION:
                last_tstep = time.time()

                # save images
                images.append(obs["image_primary"][-1])
                goals.append(goal_image)

                if FLAGS.show_image:
                    bgr_img = cv2.cvtColor(obs["image_primary"][-1], cv2.COLOR_RGB2BGR)
                    cv2.imshow("img_view", bgr_img)
                    cv2.waitKey(20)

                # get action
                forward_pass_time = time.time()
                action = np.array(policy_fn(obs, task), dtype=np.float64)
                print("forward pass time: ", time.time() - forward_pass_time)

                # perform environment step
                start_time = time.time()
                obs, _, _, truncated, _ = env.step(action)
                print("step time: ", time.time() - start_time)

                t += 1

                if truncated:
                    break

        # save video
        if FLAGS.video_save_path is not None:
            os.makedirs(FLAGS.video_save_path, exist_ok=True)
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                FLAGS.video_save_path,
                f"{curr_time}.mp4",
            )
            video = np.concatenate([np.stack(goals), np.stack(images)], axis=1)
            imageio.mimsave(save_path, video, fps=1.0 / STEP_DURATION * 3)


if __name__ == "__main__":
    app.run(main)
