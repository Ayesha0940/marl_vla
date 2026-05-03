"""
    python configs/runtime/run_eval_from_config.py \
    --config /home/axs0940/marl_vla/configs/runtime/diffusion_policy_lift_20260430_002247.json \
    --ckpt /home/axs0940/marl_vla/checkpoints/bc_diffusion_lift/bc_diffusion_lift_20260430_002247/20260430002250/models/model_epoch_500.pth \
    --name eval_lift_diffusion_best \
    --n_rollouts 50 \
    --seed 42

"""


import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained robomimic agent using its training config."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the base JSON config file used for training.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pth file) to evaluate.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="A name for this evaluation run. Results will be saved in a sub-directory. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=50,
        help="Number of rollout episodes to run.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=400,
        help="Horizon for each rollout episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed for evaluation rollouts."
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="If specified, render rollouts to screen.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="If specified, save rollout videos to the evaluation directory.",
    )

    args = parser.parse_args()

    # --- 1. Load and modify the configuration ---
    print(f"Loading base config from: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)

    run_name = args.name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    is_custom_diffusion_policy = "model" in config and "diffusion" in config and "algo" not in config

    if is_custom_diffusion_policy:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        eval_script = os.path.join(project_root, "evaluation", "eval_diffusion_policy.py")
        command = [
            "python",
            eval_script,
            "--agent",
            os.path.abspath(args.ckpt),
            "--n_rollouts",
            str(args.n_rollouts),
            "--horizon",
            str(args.horizon),
            "--seed",
            str(args.seed),
        ]

        print(f"\nExecuting command:\n{' '.join(command)}\n")
        subprocess.run(command, check=True)
        return

    # Point to the checkpoint to load
    config["experiment"]["ckpt_path"] = os.path.abspath(args.ckpt)
    config["experiment"]["name"] = run_name

    # Modify the output directory to avoid overwriting training files
    base_output_dir = config["train"]["output_dir"]
    eval_output_dir = os.path.join(base_output_dir, "evaluations")
    config["train"]["output_dir"] = eval_output_dir
    print(f"Evaluation results will be saved to: {os.path.join(eval_output_dir, run_name)}")

    # Enable rollouts
    config["experiment"]["rollout"]["enabled"] = True
    config["experiment"]["rollout"]["n"] = args.n_rollouts
    config["experiment"]["rollout"]["horizon"] = args.horizon

    # Set evaluation seed
    config["train"]["seed"] = args.seed

    # Handle rendering and video saving
    config["experiment"]["render"] = args.render
    config["experiment"]["render_video"] = args.video

    # Disable training-specific features to ensure we only run evaluation
    config["train"]["num_epochs"] = 0
    config["experiment"]["validate"] = False
    config["experiment"]["save"]["enabled"] = False

    # --- 2. Save the modified config to a temporary file ---
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json", dir=".") as tmp_f:
        json.dump(config, tmp_f, indent=4)
        tmp_config_path = tmp_f.name
    
    print(f"Saved temporary evaluation config to: {tmp_config_path}")

    # --- 3. Construct and run the evaluation command ---
    # --- 3. Construct and run the evaluation command ---
    # All remaining configs keep the legacy robomimic dispatch.
    command = ["python", "-m", "robomimic.scripts.run_trained_agent"]
    command.extend([
        "--agent", os.path.abspath(args.ckpt),
        "--n_rollouts", str(args.n_rollouts),
        "--horizon", str(args.horizon),
        "--seed", str(args.seed)
    ])

    if args.video:
        video_path = os.path.join(eval_output_dir, run_name, "eval_video.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        command.extend(["--video_path", video_path])

    print(f"\nExecuting command:\n{' '.join(command)}\n")

    try:
        subprocess.run(command, check=True)
    finally:
        os.remove(tmp_config_path)
        print(f"\nCleaned up temporary config file: {tmp_config_path}")

if __name__ == "__main__":
    main()