"""Shared launcher metadata and helpers for evaluation entrypoints."""

from typing import Dict, List, Optional, Tuple


# (task, suite) -> script path under evaluation/
SCRIPT_MAP: Dict[Tuple[str, str], str] = {
    ("can", "standard"): "__internal__",
    ("can", "filters"): "__internal__",
    ("can", "robustness"): "__internal__",
    ("can", "robustness-diffusion"): "__internal__",
    ("can", "robustness-video"): "__internal__",
    ("lift", "standard"): "__internal__",
    ("lift", "filters"): "__internal__",
    ("lift", "kalman"): "__internal__",
    ("lift", "robustness"): "__internal__",
    ("lift", "robustness-diffusion"): "__internal__",
    ("lift", "robustness-video"): "__internal__",
    ("square", "filters"): "__internal__",
    ("square", "filters-transformer"): "__internal__",
    ("square", "robustness-diffusion"): "__internal__",
    ("square", "robustness-video"): "__internal__",
    ("transport", "standard"): "__internal__",
    ("transport", "sweep"): "__internal__",
}

# Common optional flags: arg name -> CLI flag.
COMMON_FORWARD_FLAGS: List[Tuple[str, str]] = [
    ("checkpoint_path", "--checkpoint_path"),
    ("checkpoint_root", "--checkpoint_root"),
    ("run_dir", "--run_dir"),
    ("diffusion_model", "--diffusion_model"),
    ("epoch", "--epoch"),
    ("n_rollouts", "--n_rollouts"),
    ("horizon", "--horizon"),
    ("seed", "--seed"),
    ("render_gpu_id", "--render_gpu_id"),
]

# Bool switches forwarded only when True.
COMMON_BOOL_SWITCHES: List[Tuple[str, str]] = [
    ("best", "--best"),
    ("all_runs", "--all_runs"),
]

# Multi-value options
COMMON_MULTI_FLAGS: List[Tuple[str, str]] = [
    ("t_start", "--t_start"),
]


def suites_for_task(task: str) -> List[str]:
    return sorted(s for (t, s) in SCRIPT_MAP if t == task)


def all_tasks() -> List[str]:
    return sorted({t for (t, _) in SCRIPT_MAP})


def build_command(args, target_script: str, python_executable: str) -> List[str]:
    cmd: List[str] = [python_executable, target_script]

    for attr, flag in COMMON_FORWARD_FLAGS:
        value = getattr(args, attr)
        if value is not None:
            cmd.extend([flag, str(value)])

    for attr, flag in COMMON_BOOL_SWITCHES:
        if getattr(args, attr):
            cmd.append(flag)

    for attr, flag in COMMON_MULTI_FLAGS:
        values = getattr(args, attr)
        if values:
            cmd.append(flag)
            cmd.extend(str(v) for v in values)

    return cmd


def validate_task_suite(task: Optional[str], suite: Optional[str]) -> Optional[str]:
    if not task or not suite:
        return "Both --task and --suite are required unless --list is used."

    key = (task, suite)
    if key not in SCRIPT_MAP:
        available = ", ".join(suites_for_task(task)) if task in all_tasks() else ""
        if available:
            return f"Unsupported suite '{suite}' for task '{task}'. Available: {available}"
        return f"Unsupported task '{task}'."

    return None


def print_supported_pairs() -> None:
    print("Supported evaluations:")
    for task in all_tasks():
        print(f"  {task}:")
        for suite in suites_for_task(task):
            print(f"    - {suite}")
