#!/usr/bin/env python
"""
Generate a BC-RNN config for the transport task.

Usage:
  python training/gen_config_transport.py
  python training/gen_config_transport.py --output configs/bc_rnn_transport_custom.json
  python training/gen_config_transport.py --overwrite
"""

import argparse
import json
import os
import sys

from robomimic.config import config_factory


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate transport BC-RNN config")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "bc_rnn_transport.json"),
        help="Path to output config JSON",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists",
    )
    return parser.parse_args()


def build_config():
    cfg = config_factory("bc")

    # Experiment
    cfg.experiment.name = "bc_rnn_transport_gmm"
    cfg.experiment.validate = True
    cfg.experiment.rollout.enabled = True
    cfg.experiment.rollout.n = 100
    cfg.experiment.rollout.horizon = 700
    cfg.experiment.rollout.rate = 10
    cfg.experiment.rollout.warmstart = 0
    cfg.experiment.rollout.terminate_on_success = True
    cfg.experiment.render_video = False

    cfg.experiment.save.enabled = True
    cfg.experiment.save.every_n_epochs = 25
    cfg.experiment.save.on_best_validation = False
    cfg.experiment.save.on_best_rollout_return = False
    cfg.experiment.save.on_best_rollout_success_rate = True

    # Training
    cfg.train.data = os.path.join(PROJECT_ROOT, "datasets", "transport", "ph", "low_dim_v141.hdf5")
    cfg.train.output_dir = os.path.join(PROJECT_ROOT, "checkpoints", "bc_rnn", "bc_rnn_transport_gmm")
    cfg.train.num_data_workers = 0
    cfg.train.hdf5_cache_mode = "all"
    cfg.train.hdf5_use_swmr = True
    cfg.train.hdf5_load_next_obs = False
    cfg.train.hdf5_normalize_obs = False
    cfg.train.hdf5_filter_key = "train"
    cfg.train.hdf5_validation_filter_key = "valid"

    cfg.train.seq_length = 16
    cfg.train.pad_seq_length = True
    cfg.train.frame_stack = 1
    cfg.train.pad_frame_stack = True
    cfg.train.dataset_keys = ["actions", "rewards", "dones"]

    cfg.train.batch_size = 32
    cfg.train.num_epochs = 1000
    cfg.train.seed = 1

    # Policy / algo
    cfg.algo.actor_layer_dims = [400, 400]

    # Enable multimodal action modeling for long-horizon two-arm transport.
    cfg.algo.gmm.enabled = True
    cfg.algo.gmm.num_modes = 10
    cfg.algo.gmm.min_std = 0.01

    cfg.algo.rnn.enabled = True
    cfg.algo.rnn.horizon = 16
    cfg.algo.rnn.hidden_dim = 400
    cfg.algo.rnn.rnn_type = "LSTM"
    cfg.algo.rnn.num_layers = 2
    cfg.algo.rnn.open_loop = False
    cfg.algo.rnn.kwargs.bidirectional = False

    cfg.algo.transformer.enabled = False

    # Observations
    cfg.observation.modalities.obs.low_dim = [
        "robot0_joint_pos",
        "robot0_joint_pos_cos",
        "robot0_joint_pos_sin",
        "robot0_joint_vel",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_eef_vel_lin",
        "robot0_eef_vel_ang",
        "robot0_gripper_qpos",
        "robot0_gripper_qvel",
        "robot1_joint_pos",
        "robot1_joint_pos_cos",
        "robot1_joint_pos_sin",
        "robot1_joint_vel",
        "robot1_eef_pos",
        "robot1_eef_quat",
        "robot1_eef_vel_lin",
        "robot1_eef_vel_ang",
        "robot1_gripper_qpos",
        "robot1_gripper_qvel",
        "object",
    ]

    return cfg


def main():
    args = parse_args()

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(PROJECT_ROOT, output_path)

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_path) and not args.overwrite:
        print(f"Config already exists at {output_path}")
        print("Use --overwrite to replace it.")
        return 0

    cfg = build_config()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)

    print(f"Config saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
