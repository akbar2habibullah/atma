import os
import sys
import multiprocessing as mp
from pathlib import Path
from huggingface_hub import hf_hub_download

import numpy as np
import torch
from tqdm import tqdm

def get_data(fname, repo_id="kjj0/finewebedu10B-gpt2"):
    local_dir = os.path.join(os.getcwd(), 'finewebedu10B')
    if not os.path.exists(os.path.join(local_dir, fname)):
        hf_hub_download(repo_id=repo_id, filename=fname,
                        repo_type="dataset", local_dir=local_dir)

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    token_bytes = int(header[3]) or 2  # 0 = legacy file, default to uint16
    assert token_bytes in (2, 4), f"unsupported token width in header: {token_bytes}"
    # torch has no uint32; use int32 (same width, token ids are positive so no sign issue)
    torch_dtype = torch.uint16 if token_bytes == 2 else torch.int32
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch_dtype, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == token_bytes * num_tokens, "number of tokens read does not match header"
    return tokens

def write_datafile(filename: str, toks, use_uint32: bool = False):
    """
    Saves token data as a .bin file compatible with _load_data_shard.
    Header: 256 x int32 — [magic, version, token_count, bytes_per_token, zeros...].
    Body: tokens as uint16 (default) or uint32.
    """
    assert len(toks) < 2**31, "token count too large"
    np_dtype = np.uint32 if use_uint32 else np.uint16
    token_bytes = 4 if use_uint32 else 2
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520
    header[1] = 1
    header[2] = len(toks)
    header[3] = token_bytes
    if not isinstance(toks, np.ndarray) or toks.dtype != np_dtype:
        max_id = 2**32 if use_uint32 else 2**16
        assert all(0 <= t < max_id for t in toks), f"token id too large for {'uint32' if use_uint32 else 'uint16'}"
        toks = np.array(toks, dtype=np_dtype)
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks.tobytes())


def tokenize_to_bin(
    dataset_name: str,
    tokenizer_name: str,
    output_dir: str,
    file_prefix: str = "data",
    shard_size: int = 10**8,
    text_field: str = "text",
    split: str = "train",
    dataset_config: str = None,
    num_proc: int = None,
):
    """
    Tokenizes a HuggingFace dataset with a chosen HF tokenizer and writes
    shards as .bin files readable by _load_data_shard / data_generator.

    The first shard is saved as a validation split; remaining shards are train.

    Args:
        dataset_name:    HF dataset repo id, e.g. "HuggingFaceFW/fineweb"
        tokenizer_name:  HF tokenizer repo id, e.g. "gpt2" or "meta-llama/Llama-2-7b-hf"
        output_dir:      Directory where .bin shards will be written
        file_prefix:     Filename prefix for output shards
        shard_size:      Number of tokens per shard (default 100M)
        text_field:      Column in dataset that holds the document text
        split:           Dataset split to load (default "train")
        dataset_config:  Optional dataset config name passed to load_dataset
        num_proc:        Worker processes for tokenisation (default: cpu_count - 2)
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # Resolve EOS token id to use as document delimiter
    eot_id = tokenizer.eos_token_id
    if eot_id is None:
        raise ValueError(f"Tokenizer {tokenizer_name!r} has no eos_token_id")

    print(f"Loading dataset: {dataset_name} (config={dataset_config}, split={split})")
    ds = load_dataset(dataset_name, name=dataset_config, split=split)

    use_uint32 = tokenizer.vocab_size > 2**16
    np_dtype = np.uint32 if use_uint32 else np.uint16
    max_id = 2**32 if use_uint32 else 2**16
    print(f"Vocab size {tokenizer.vocab_size} → using {'uint32' if use_uint32 else 'uint16'} tokens")

    def _tokenize(doc):
        ids = [eot_id] + tokenizer.encode(doc[text_field], add_special_tokens=False)
        arr = np.array(ids, dtype=np.uint64)
        assert (arr < max_id).all(), "token id exceeds storage dtype range"
        return arr.astype(np_dtype)

    nprocs = num_proc if num_proc is not None else max(1, os.cpu_count() - 2)
    shard_index = 0
    all_tokens = np.empty((shard_size,), dtype=np_dtype)
    token_count = 0
    progress_bar = None

    with mp.Pool(nprocs) as pool:
        for tokens in pool.imap(_tokenize, ds, chunksize=16):
            if token_count + len(tokens) < shard_size:
                all_tokens[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens",
                                        desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                split_label = "val" if shard_index == 0 else "train"
                fname = os.path.join(output_dir,
                                     f"{file_prefix}_{split_label}_{shard_index:06d}.bin")
                remainder = shard_size - token_count
                if progress_bar is not None:
                    progress_bar.update(remainder)
                    progress_bar.close()
                all_tokens[token_count:token_count + remainder] = tokens[:remainder]
                write_datafile(fname, all_tokens, use_uint32=use_uint32)
                shard_index += 1
                progress_bar = None
                leftover = len(tokens) - remainder
                all_tokens[:leftover] = tokens[remainder:]
                token_count = leftover

    if token_count > 0:
        split_label = "val" if shard_index == 0 else "train"
        fname = os.path.join(output_dir,
                             f"{file_prefix}_{split_label}_{shard_index:06d}.bin")
        write_datafile(fname, all_tokens[:token_count], use_uint32=use_uint32)


def data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    files = sorted(Path.cwd().glob(filename_pattern))
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos : pos + batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)