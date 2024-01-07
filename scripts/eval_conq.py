from copy import copy

import rerun as rr
import time
from typing import Optional, Tuple

import bosdyn.client
import cv2
import gym
import jax
import numpy as np
import rerun as rr
from absl import app, flags, logging
from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseKeepAlive, LeaseClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.math_helpers import SE3Pose, Quat, Vec3
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
flags.DEFINE_integer("exec_horizon", 10, "Length of action sequence to execute")

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
        obs = {
            'image_primary': image_primary,
            'proprio': prioprio,
            # NOTE: we get a warning if we don't pad the mask, but it seems to work fine without it,
            #  and trying to add it causes other issues
            # 'pad_mask_dict': {
            #     'image_primary': np.ones((self.im_size, self.im_size)),
            #     'proprio': np.ones((66,)),
            # }
        }
        return obs

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        t0 = time.time()
        cmd_with_gripper = self.model_action_to_conq_cmd(action)

        # Execute!
        self.clients.command.robot_command(cmd_with_gripper)

        obs = self.get_obs()
        t1 = time.time()
        print(f"one step time {t1 - t0:.3f}")

        # FIXME: how to set trunc? Are we sure it should be the same as "is_terminal" in the action?
        trunc = False
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
        open_fraction = action[6]
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

    ##################################
    # DEBUGGING
    ##################################
    from octo.data.oxe import make_oxe_dataset_kwargs
    from octo.data.dataset import make_single_dataset
    from data.utils.data_utils import NormalizationType
    dataset_action_horizon = 500
    dataset = make_single_dataset(
        dataset_kwargs=dict(
            name='conq_hose_manipulation:1.2.0',
            data_dir="/home/peter/tensorflow_datasets",
            image_obs_keys={"primary": "hand_color_image"},
            state_obs_keys=["state"],
            language_key="language_instruction",
            action_proprio_normalization_type=NormalizationType.NORMAL,
            absolute_action_mask=[False, False, False, False, False, False, True, True],
        ),
        traj_transform_kwargs=dict(
            window_size=1,
            future_action_window_size=dataset_action_horizon - 1,
        ),
        frame_transform_kwargs=dict(
            resize_size={"primary": (256, 256)},
        ),
        train=False,  # use the validation dataset
    )
    iterator = list(dataset.unbatch().iterator())

    with (LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)):
        look_cmd = hand_pose_cmd(clients, 0.5, 0, 0.4, 0, np.deg2rad(75), 0, duration=1)
        blocking_arm_command(clients, look_cmd)
        open_gripper(clients)

        state = env.clients.state.get_robot_state()
        snapshot = state.kinematic_state.transforms_snapshot
        hand_in_vision0 = get_a_tform_b(snapshot, VISION_FRAME_NAME, HAND_FRAME_NAME)
        body_in_vision0 = get_a_tform_b(snapshot, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
        hand_in_body0 = get_a_tform_b(snapshot, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME)

        rr.set_time_sequence('t', 0)
        viz_common_frames(snapshot)
        i = 0
        debug_batch = iterator[i]
        example_actions = debug_batch['action']
        hand_in_vision = copy(hand_in_vision0)
        for t, action_normalized in enumerate(example_actions):
            action = env.unnormalize(action_normalized, env.action_proprio_metadata["action"])
            rr.set_time_sequence('t', t)
            rr.log("d/x", rr.TimeSeriesScalar(action[0]))
            rr.log("d/z", rr.TimeSeriesScalar(action[2]))
            rr.log("gripper/open_fraction", rr.TimeSeriesScalar(action[6]))
            # Compute hand to body by subtracting the body_in_vision0 from the current hand_in_vision
            hand_in_body = body_in_vision0.inverse() * hand_in_vision
            new_hand_in_vision = env.get_new_hand_in_vision(action, body_in_vision0, hand_in_body)
            rr_tform(f'hand_pred', new_hand_in_vision)
            hand_in_vision = new_hand_in_vision
        print("done")
    return
    ##################################

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
