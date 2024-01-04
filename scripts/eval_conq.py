import os
import rerun as rr
import time
from datetime import datetime
from typing import Optional, Tuple

import bosdyn.client
import cv2
import gym
import imageio
import jax
import numpy as np
from absl import app, flags, logging
from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseKeepAlive, LeaseClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.math_helpers import quat_to_eulerZYX, SE3Velocity, SE3Pose, Quat, Vec3
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.robot_state import RobotStateClient
from conq.cameras_utils import get_color_img
from conq.clients import Clients
from conq.data_recorder import get_state_vec
from conq.hand_motion import hand_pose_cmd
from conq.manipulation import open_gripper, add_follow_with_body, blocking_arm_command
from conq.utils import setup_and_stand
from gym.core import ObsType, ActType
from scipy.spatial.transform import Rotation as Rot

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import add_octo_env_wrappers

np.set_printoptions(suppress=True, precision=4)

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
flags.DEFINE_integer("exec_horizon", 3, "Length of action sequence to execute")

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


def my_euler_to_quat(euler):
    q_xyzw = Rot.from_euler("xyz", euler, degrees=False).as_quat().tolist()
    q_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
    return q_wxyz


def my_quat_to_euler(q):
    euler = Rot.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz", degrees=False).tolist()
    return euler


def viz_common_frames(snapshot):
    body_in_vision = get_a_tform_b(snapshot, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
    hand_in_vision = get_a_tform_b(snapshot, VISION_FRAME_NAME, HAND_FRAME_NAME)
    rr_tform('body', body_in_vision)
    rr_tform('hand', hand_in_vision)
    rr.log(f'frames/vision', rr.Transform3D(translation=[0, 0, 0]))


def rr_tform(child_frame: str, tform: SE3Pose):
    translation = np.array([tform.position.x, tform.position.y, tform.position.z])
    rot_mat = tform.rotation.to_matrix()
    rr.log(f"frames/{child_frame}", rr.Transform3D(rr.TranslationAndMat3x3(translation, rot_mat)))


class ConqGymEnv(gym.Env):

    def __init__(self, clients: Clients, im_size, blocking):
        self.im_size = im_size
        self.blocking = blocking
        self.clients = clients

        self.observation_space = gym.spaces.Dict({
            # TODO: add proprio
            'image_primary': gym.spaces.Box(low=0, high=255, shape=(self.im_size, self.im_size, 3), dtype=np.uint8),
            'proprio': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(66,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        obs = self.get_obs()
        return obs, {}

    def get_obs(self):
        state = self.clients.state.get_robot_state()
        prioprio = get_state_vec(state)
        image_primary, _ = get_color_img(self.clients.image, "hand_color_image")
        # TODO: resize img to im_size,im_size?
        obs = {
            'image_primary': image_primary,
            'proprio': prioprio,
        }
        return obs

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        t0 = time.time()
        # Action is an 8 dim vector, and if you look at the features.json you'll find its meaning:
        # [ dx, dy, dz, droll, dpitch, dyaw, open_fraction, is_terminal ]
        # where the delta position and orientation is in the current body frame
        # First we're going to take the deltas in current body frame, and transform those into deltas in vision frame
        delta_pose_in_body = SE3Pose(
            x=action[0],
            y=action[1],
            z=action[2],
            rot=Quat(*my_euler_to_quat([action[3], action[4], action[5]])),
        )

        state = self.clients.state.get_robot_state()
        snapshot = state.kinematic_state.transforms_snapshot
        hand_in_vision = get_a_tform_b(snapshot, VISION_FRAME_NAME, HAND_FRAME_NAME)
        body_in_vision = get_a_tform_b(snapshot, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

        # The delta pose just needs to be rotated
        delta_pose_in_vision_rot = body_in_vision.rotation * delta_pose_in_body.rotation
        delta_pose_in_vision_trans = body_in_vision.rotation * Vec3.from_proto(delta_pose_in_body.position)
        delta_pose_in_vision = SE3Pose(delta_pose_in_vision_trans.x,
                                       delta_pose_in_vision_trans.y,
                                       delta_pose_in_vision_trans.z,
                                       delta_pose_in_vision_rot)
        logging.debug(delta_pose_in_body)
        logging.debug(delta_pose_in_vision)

        new_hand_in_vision = delta_pose_in_vision * hand_in_vision


        arm_cmd = RobotCommandBuilder.arm_pose_command_from_pose(new_hand_in_vision.to_proto(), VISION_FRAME_NAME,
                                                                 seconds=2 * STEP_DURATION)

        cmd = add_follow_with_body(arm_cmd)
        cmd.synchronized_command.arm_command.arm_cartesian_command.max_linear_velocity.value = 0.3

        # Add gripper command
        open_fraction = action[6]
        # FIXME: limiting for initial testing!!!
        open_fraction = np.clip(open_fraction, 0.1, 0.9)

        gripper_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(open_fraction)
        cmd_with_gripper = RobotCommandBuilder.build_synchro_command(cmd, gripper_cmd)
        logging.debug(cmd_with_gripper)

        # rerun logging
        viz_common_frames(snapshot)
        rr_tform('current_hand', hand_in_vision)
        rr_tform('new_hand', new_hand_in_vision)
        rr.log("gripper/open_fraction", open_fraction)

        # Execute!
        self.clients.command.robot_command(cmd_with_gripper)

        obs = self.get_obs()
        t1 = time.time()
        print(f"one step time {t1 - t0:.3f}")

        # FIXME: how to set trunc? Are we sure it should be the same as "is_terminal" in the action?
        trunc = False
        return obs, 0.0, False, trunc, {}


def main(_):
    # setup Conq
    sdk = bosdyn.client.create_standard_sdk('EvalClient')
    robot = sdk.create_robot("192.168.80.3")
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    rr.init('eval')
    rr.connect()

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)

    lease_client.take()

    command_client = setup_and_stand(robot)

    clients = Clients(lease=lease_client, state=robot_state_client, manipulation=manipulation_api_client,
                      image=image_client, raycast=None, command=command_client, robot=robot, recorder=None)

    # load models
    model = OctoModel.load_pretrained(
        FLAGS.checkpoint_weights_path,
        FLAGS.checkpoint_step,
    )

    # wrap the robot environment
    env = ConqGymEnv(clients, FLAGS.im_size, FLAGS.blocking)
    env = add_octo_env_wrappers(env, model.config, dict(model.dataset_statistics), normalization_type="normal",
                                resize_size=(FLAGS.im_size, FLAGS.im_size), exec_horizon=FLAGS.exec_horizon)

    rng = jax.random.PRNGKey(1)
    goal_instruction = "grasp hose"

    with (LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)):
        look_cmd = hand_pose_cmd(clients, 0.5, 0, 0.4, 0, np.deg2rad(75), 0, duration=1)
        blocking_arm_command(clients, look_cmd)
        open_gripper(clients)

        while True:
            task = model.create_tasks(texts=[goal_instruction])

            # reset env
            obs, _ = env.reset()
            time.sleep(2.0)

            # input("Press [Enter] to start.")

            # do rollout
            last_tstep = time.time()
            images = []
            t = 0
            while t < FLAGS.num_timesteps:
                if time.time() > last_tstep + STEP_DURATION:
                    last_tstep = time.time()

                    # save images
                    images.append(obs["image_primary"][-1])

                    if FLAGS.show_image:
                        bgr_img = cv2.cvtColor(obs["image_primary"][-1], cv2.COLOR_RGB2BGR)
                        cv2.imshow("img_view", bgr_img)
                        cv2.waitKey(20)

                    # get action
                    forward_pass_time = time.time()

                    actions = model.sample_actions(jax.tree_map(lambda x: x[None], obs), task, rng=rng)
                    actions = actions[0]
                    print("forward pass time: ", time.time() - forward_pass_time)

                    # perform environment step
                    start_time = time.time()
                    obs, _, _, truncated, _ = env.step(actions)
                    print("step time: ", time.time() - start_time)

                    t += 1

                    if truncated:
                        break

            # save video
            # if FLAGS.video_save_path is not None:
            #     os.makedirs(FLAGS.video_save_path, exist_ok=True)
            #     curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            #     save_path = os.path.join(
            #         FLAGS.video_save_path,
            #         f"{curr_time}.mp4",
            #     )
            #     video = np.concatenate([np.stack(images)], axis=1)
            #     imageio.mimsave(save_path, video, fps=1.0 / STEP_DURATION * 3)


if __name__ == "__main__":
    app.run(main)
