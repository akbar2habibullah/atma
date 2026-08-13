"""Download and verify the token shards needed by the Foveal sweep."""

from __future__ import annotations

import argparse
import glob
import struct
from pathlib import Path

from .config import FovealConfig


MAGIC = 20240520
VERSION = 1


def shard_token_count(path: str | Path) -> int:
    path = Path(path)
    with path.open("rb") as handle:
        header = handle.read(16)
    if len(header) != 16:
        raise ValueError(f"token shard header is truncated: {path}")
    magic, version, token_count, token_bytes = struct.unpack("<4i", header)
    if magic != MAGIC:
        raise ValueError(f"token shard has invalid magic number: {path}")
    if version != VERSION:
        raise ValueError(f"token shard has unsupported version {version}: {path}")
    token_bytes = token_bytes or 2
    if token_bytes not in (2, 4):
        raise ValueError(f"token shard has invalid token width {token_bytes}: {path}")
    expected_size = 256 * 4 + token_count * token_bytes
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"token shard size mismatch for {path}: expected {expected_size}, got {actual_size}"
        )
    return token_count


def usable_training_tokens(paths: list[Path], batch_tokens: int) -> int:
    """Tokens consumable by TokenShardLoader without crossing shard boundaries."""

    return sum(
        max(0, (shard_token_count(path) - 2) // batch_tokens) * batch_tokens
        for path in paths
    )


def _download(config: FovealConfig, filename: str, local_dir: Path) -> Path:
    if config.dataset_repo is None:
        raise FileNotFoundError(
            f"missing dataset file {local_dir / filename}; dataset_repo is disabled"
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download the configured token dataset"
        ) from exc
    print(f"[data] downloading {config.dataset_repo}/{filename}", flush=True)
    return Path(
        hf_hub_download(
            repo_id=config.dataset_repo,
            filename=filename,
            repo_type="dataset",
            local_dir=local_dir,
            cache_dir=config.hf_cache,
        )
    )


def ensure_validation_data(config: FovealConfig, pattern: str | None = None) -> list[Path]:
    pattern = pattern or str(Path(config.train_glob).parent / "finewebedu_val_*.bin")
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not paths and config.auto_download_data:
        local_dir = Path(config.train_glob).parent
        _download(config, config.dataset_validation_file, local_dir)
        paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"no validation token shards match {pattern!r}")
    for path in paths:
        shard_token_count(path)
    return paths


def ensure_training_data(config: FovealConfig, *, include_validation: bool = True) -> list[Path]:
    """Ensure enough sequential shards exist for the configured optimizer steps."""

    target_tokens = config.train_steps * config.batch_tokens
    local_dir = Path(config.train_glob).parent
    paths: list[Path] = []
    available = 0
    index = 1
    while available < target_tokens:
        filename = config.dataset_train_template.format(index=index)
        path = local_dir / filename
        if not path.is_file():
            if not config.auto_download_data:
                raise FileNotFoundError(
                    f"missing sequential dataset shard {path}; "
                    f"the run requires {target_tokens:,} usable tokens"
                )
            path = _download(config, filename, local_dir)
        token_count = shard_token_count(path)
        paths.append(path)
        available += max(0, (token_count - 2) // config.batch_tokens) * config.batch_tokens
        index += 1
        if index > 100_000:
            raise RuntimeError("dataset shard search exceeded 100,000 files")

    matched = {Path(item).resolve() for item in glob.glob(config.train_glob)}
    unmatched = [path for path in paths if path.resolve() not in matched]
    if unmatched:
        raise ValueError(
            f"downloaded shards do not match train_glob={config.train_glob!r}: {unmatched}"
        )

    if include_validation:
        ensure_validation_data(config)
    print(
        f"[data] ready repo={config.dataset_repo or 'local'} shards={len(paths)} "
        f"usable_tokens={available:,} required={target_tokens:,}",
        flush=True,
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FovealConfig.load(args.config)
    ensure_training_data(config, include_validation=not args.no_validation)


if __name__ == "__main__":
    main()
