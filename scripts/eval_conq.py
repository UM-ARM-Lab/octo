#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
from typing import Optional, Tuple, Dict

import bosdyn.client
import gym
import jax
import numpy as np
import rerun as rr
from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.lease import LeaseKeepAlive, LeaseClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.math_helpers import SE3Pose, Quat
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.robot_state import RobotStateClient
from conq.cameras_utils import get_color_img, source_to_fmt, image_to_opencv
from conq.clients import Clients
from conq.data_recorder import get_state_vec
from conq.hand_motion import hand_pose_cmd
from conq.manipulation import open_gripper, blocking_arm_command
from conq.utils import setup_and_stand
from gym.core import ObsType, ActType
from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import add_octo_env_wrappers
from scipy.spatial.transform import Rotation as Rot

np.set_printoptions(suppress=True, precision=4)

# Derived from the mean frequency in the training dataset.
# see `check_control_rate` in `preprocesser.py` in `conq_hose_manipulation_dataset_builder`.
STEP_DURATION = 1 / 7.7


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

    def __init__(self, clients: Clients, im_size, model_dataset_kwargs: Dict):
        self.im_size = im_size
        self.clients = clients
        self.model_kwargs = model_dataset_kwargs
        self.image_obs_keys = self.model_kwargs['image_obs_keys']
        self.use_proprio = self.model_kwargs['state_obs_keys'] is not None

        obs_space_dict = {}
        self.img_srcs = []
        self.img_fmts = []
        for image_obs_key, img_src in self.image_obs_keys.items():
            if img_src is not None:
                obs_space_dict[image_obs_key] = gym.spaces.Box(low=0, high=255, shape=(self.im_size, self.im_size, 3),
                                                               dtype=np.uint8)
                self.img_srcs.append(img_src)
                self.img_fmts.append(source_to_fmt(img_src))

        if self.use_proprio:
            obs_space_dict["proprio"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(66,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict(obs_space_dict)
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        # for tracking the control frequency
        self.dts = []

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
        t2 = time.time()
        print(f"getting imgs {t2 - t1:.3f}")

        obs = {}
        for image_obs_key, img_np in zip(self.image_obs_keys, imgs):
            obs[image_obs_key] = img_np

        if self.use_proprio:
            state = self.clients.state.get_robot_state()
            prioprio = get_state_vec(state)
            obs["proprio"] = prioprio

        return obs

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        t0 = time.time()
        cmd_with_gripper = self.model_action_to_conq_cmd(action)

        # Execute!
        self.clients.command.robot_command(cmd_with_gripper)

        obs = self.get_obs()

        # FIXME: how to set trunc? Are we sure it should be the same as "is_terminal" in the action?
        trunc = action[-1] > 0.95

        # Sleep if we're going too fast
        sleep_dt = STEP_DURATION - (time.time() - t0)
        if sleep_dt > 0:
            time.sleep(sleep_dt)

        # Log control frequency
        t1 = time.time()
        self.dts.append(t1 - t0)
        freq = 1 / np.mean(self.dts)
        print(f"control freq: {freq:.3f}Hz")
        if len(self.dts) > 10:  # circular buffer
            self.dts.pop(0)
        return obs, 0.0, False, trunc, {}

    def model_action_to_conq_cmd(self, action):
        state = self.clients.state.get_robot_state()
        snapshot = state.kinematic_state.transforms_snapshot
        hand_in_vision = get_a_tform_b(snapshot, VISION_FRAME_NAME, HAND_FRAME_NAME)
        body_in_vision = get_a_tform_b(snapshot, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
        hand_in_body = get_a_tform_b(snapshot, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME)

        new_hand_in_vision = self.get_new_hand_in_vision(action, body_in_vision, hand_in_body)

        arm_cmd = RobotCommandBuilder.arm_pose_command_from_pose(new_hand_in_vision.to_proto(), VISION_FRAME_NAME,
                                                                 seconds=2 * STEP_DURATION)
        cmd = arm_cmd
        # cmd = add_follow_with_body(arm_cmd)
        # cmd.synchronized_command.arm_command.arm_cartesian_command.max_linear_velocity.value = 0.3

        # Add gripper command
        open_fraction = np.clip(action[6], 0, 1)
        gripper_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(open_fraction)
        cmd_with_gripper = RobotCommandBuilder.build_synchro_command(cmd, gripper_cmd)

        # rerun logging
        viz_common_frames(snapshot)
        rr_tform('current_hand', hand_in_vision)
        rr_tform('new_hand', new_hand_in_vision)
        rr.log("gripper/open_fraction", rr.TimeSeriesScalar(open_fraction))
        return cmd_with_gripper

    @staticmethod
    def get_new_hand_in_vision(action, body_in_vision, hand_in_body):
        # Action is an 8 dim vector, and if you look at the features.json you'll find its meaning:
        # [ dx, dy, dz, droll, dpitch, dyaw, open_fraction, is_terminal ]
        # where the delta position and orientation is in the current body frame
        # First we're going to take the deltas in current body frame,
        # and apply it to the current hand in body frame to get the new hand in body frame
        delta_pose_in_body = SE3Pose(x=action[0], y=action[1], z=action[2],
                                     rot=Quat(*my_euler_to_quat([action[3], action[4], action[5]])))
        new_hand_in_body = delta_pose_in_body * hand_in_body
        # Now we need to transform the new hand in body frame into the new hand in vision frame
        new_hand_in_vision = body_in_vision * new_hand_in_body
        return new_hand_in_vision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=Path, help="path up to and including the step number")
    parser.add_argument("--num-timesteps", type=int, default=70)
    parser.add_argument("--im-size", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--exec-horizon", type=int, default=3)

    args = parser.parse_args()

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

    # load models
    checkpoint_weights_path = str(args.checkpoint_path.absolute().parent)
    checkpoint_step = int(args.checkpoint_path.name)
    model = OctoModel.load_pretrained(checkpoint_weights_path, checkpoint_step)

    lease_client.take()
    command_client = setup_and_stand(robot)
    clients = Clients(lease=lease_client, state=robot_state_client, manipulation=manipulation_api_client,
                      image=image_client, raycast=None, command=command_client, robot=robot, recorder=None)

    # wrap the robot environment
    env = ConqGymEnv(clients, args.im_size, model.config['dataset_kwargs'])
    env = add_octo_env_wrappers(env, model.config, dict(model.dataset_statistics), normalization_type="normal",
                                resize_size=(args.im_size, args.im_size), exec_horizon=args.exec_horizon)

    rng = jax.random.PRNGKey(1)
    goal_instruction = "grasp hose"

    with (LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)):

        while True:
            task = model.create_tasks(texts=[goal_instruction])

            # NOTE: should probably match the starting pose of the robot to the starting pose of the dataset
            open_gripper(clients)
            look_cmd = hand_pose_cmd(clients, 0.6, 0, 0.6, 0, np.deg2rad(0), 0, duration=0.5)
            blocking_arm_command(clients, look_cmd)
            look_cmd = hand_pose_cmd(clients, 0.8, 0, 0.2, 0, np.deg2rad(0), 0, duration=0.5)
            blocking_arm_command(clients, look_cmd)

            # reset env
            obs, _ = env.reset()

            # do rollout
            last_tstep = time.time()
            t = 0
            while t < args.num_timesteps:
                if time.time() > last_tstep + STEP_DURATION:
                    last_tstep = time.time()

                    for img_key, img_src in model.config['dataset_kwargs']['image_obs_keys'].items():
                        if img_src is not None:
                            rr.log(f"{img_key}", rr.Image(obs[img_key][-1]))

                    # get action
                    actions = model.sample_actions(jax.tree_map(lambda x: x[None], obs), task, rng=rng)
                    actions = actions[0]

                    # perform environment step
                    obs, _, _, truncated, _ = env.step(actions)

                    t += 1

                    if truncated:
                        break
            #     imageio.mimsave(save_path, video, fps=1.0 / STEP_DURATION * 3)


if __name__ == "__main__":
    main()
