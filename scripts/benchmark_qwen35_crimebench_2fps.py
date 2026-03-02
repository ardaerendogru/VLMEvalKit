import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_PY = REPO_ROOT / "run.py"


PAIRS = [
    # (model_on_gpus_0_1, model_on_gpus_2_3)
    ("Qwen3.5-0.8B-NoThinking", "Qwen3.5-2B-NoThinking"),
    ("Qwen3.5-4B-NoThinking", "Qwen3.5-9B-NoThinking"),
]


def launch_model(model_name: str, gpus: str, extra_env: dict | None = None):
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpus,
            # Skip bad / corrupted videos instead of crashing.
            "SKIP_ERR": "1",
        }
    )
    if extra_env:
        env.update(extra_env)

    cmd = [
        "python3",
        str(RUN_PY),
        "--model",
        model_name,
        "--data",
        "CrimeBench_2fps",
        "--verbose",
        "--use-vllm",
    ]
    print(f"Launching {model_name} on GPUs {gpus}: {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=env)


def main():
    for left_model, right_model in PAIRS:
        print(
            f"\n=== Benchmarking pair: {left_model} (GPU 0,1) "
            f"and {right_model} (GPU 2,3) on CrimeBench_2fps ==="
        )

        p_left = launch_model(left_model, "0,1")
        p_right = launch_model(right_model, "2,3")

        # Wait for both to finish before launching the next pair
        left_code = p_left.wait()
        right_code = p_right.wait()

        print(
            f"Finished pair {left_model} / {right_model} "
            f"with exit codes {left_code} / {right_code}"
        )


if __name__ == "__main__":
    main()

