"""Run the complete 3 checkpoint x 4 adaptation Foveal CPT matrix."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from .config import FovealConfig
from .prepare_data import ensure_training_data


CORES = ("polar", "rope", "nope")
MODES = ("local", "lm_output", "kl", "lm_output_kl")
PACKAGE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", nargs="+", choices=CORES, default=list(CORES))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def config_path(core: str, mode: str) -> Path:
    return PACKAGE_DIR / "configs" / f"{core}-{mode}.json"


def latest_checkpoint(directory: Path) -> tuple[Path, int] | None:
    latest = directory / "latest.json"
    if not latest.is_file():
        return None
    record = json.loads(latest.read_text(encoding="utf-8"))
    checkpoint = directory / record["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{latest} points to missing checkpoint {checkpoint}")
    return checkpoint, int(record["step"])


def execute(command: list[str], *, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def calibrate_core(
    core: str,
    *,
    device: str,
    smoke_steps: int | None,
    dry_run: bool,
) -> Path | None:
    path = config_path(core, "kl")
    config = FovealConfig.load(path)
    output_root = Path(config.output_dir)
    if smoke_steps is not None:
        output_root = output_root / "smoke"
    directory = output_root / "calibration"
    latest = latest_checkpoint(directory)
    target_steps = smoke_steps or math.ceil(
        config.calibration_tokens / config.calibration_batch_tokens
    )
    if latest is not None and latest[1] >= target_steps:
        print(f"[sweep] reuse {core} calibration: {latest[0]}", flush=True)
        return latest[0]

    command = [
        sys.executable,
        "-m",
        "foveal_cpt.calibrate",
        "--config",
        str(path),
        "--device",
        device,
    ]
    if latest is not None:
        command.extend(("--resume", str(latest[0])))
    if smoke_steps is not None:
        command.extend(
            ("--smoke-steps", str(smoke_steps), "--output-dir", str(output_root))
        )
    execute(command, dry_run=dry_run)
    if dry_run:
        return None
    result = latest_checkpoint(directory)
    if result is None:
        raise RuntimeError(f"calibration did not write {directory / 'latest.json'}")
    return result[0]


def run_cell(
    core: str,
    mode: str,
    *,
    calibration: Path | None,
    device: str,
    smoke_steps: int | None,
    dry_run: bool,
) -> None:
    path = config_path(core, mode)
    config = FovealConfig.load(path)
    output_root = Path(config.output_dir)
    if smoke_steps is not None:
        output_root = output_root / "smoke"
    directory = output_root / "cpt"
    latest = latest_checkpoint(directory)
    target_steps = smoke_steps or config.train_steps
    if latest is not None and latest[1] >= target_steps:
        print(f"[sweep] complete {core}/{mode}: {latest[0]}", flush=True)
        return

    command = [
        sys.executable,
        "-m",
        "foveal_cpt.train",
        "--config",
        str(path),
        "--device",
        device,
    ]
    if latest is not None:
        command.extend(("--resume", str(latest[0])))
    elif config.requires_calibration:
        if calibration is None and not dry_run:
            raise RuntimeError(f"missing calibration for {core}/{mode}")
        command.extend(("--index-checkpoint", str(calibration or "<latest-calibration>")))
    if smoke_steps is not None:
        command.extend(
            ("--smoke-steps", str(smoke_steps), "--output-dir", str(output_root))
        )
    execute(command, dry_run=dry_run)


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        ensure_training_data(FovealConfig.load(config_path(args.cores[0], args.modes[0])))
    for core in args.cores:
        calibration = None
        if any(mode in {"kl", "lm_output_kl"} for mode in args.modes):
            calibration = calibrate_core(
                core,
                device=args.device,
                smoke_steps=args.smoke_steps,
                dry_run=args.dry_run,
            )
        for mode in args.modes:
            run_cell(
                core,
                mode,
                calibration=calibration,
                device=args.device,
                smoke_steps=args.smoke_steps,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
