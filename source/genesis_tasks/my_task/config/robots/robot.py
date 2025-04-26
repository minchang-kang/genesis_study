##
# Configuration
##

ROBOT_CFG = {
    "num_actions": 12,
    # joint/link names
    "default_joint_angles": {  # [rad]
        "FL_hip_joint": 0.0,
        "FR_hip_joint": 0.0,
        "RL_hip_joint": 0.0,
        "RR_hip_joint": 0.0,
        "FL_thigh_joint": 0.8,
        "FR_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0,
        "RR_thigh_joint": 1.0,
        "FL_calf_joint": -1.5,
        "FR_calf_joint": -1.5,
        "RL_calf_joint": -1.5,
        "RR_calf_joint": -1.5,
    },
    "dof_names": [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ],
    # PD
    "kp": 20.0,
    "kd": 0.5,
    # termination
    "termination_if_roll_greater_than": 10,  # degree
    "termination_if_pitch_greater_than": 10,
    # base pose
    "base_init_pos": [0.0, 0.0, 0.42],
    "base_init_quat": [1.0, 0.0, 0.0, 0.0],
    "episode_length_s": 20.0,
    "resampling_time_s": 4.0,
    "action_scale": 0.25,
    "simulate_action_latency": True,
    "clip_actions": 100.0,
}