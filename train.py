"""
train.py

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.
"""

import os
import uuid
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F

from train.optimizer import Muon
from train.model import Model
from model.config import AtmaConfig

from train.data import get_data, data_generator

num_chunks = 1

get_data("finewebedu_val_%06d.bin" % 0)

for i in range(1, num_chunks+1):
    get_data("finewebedu_train_%06d.bin" % i)

########################################
#                Setup                 #
########################################

# Define device safely for Colab
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# logging setup
os.makedirs("logs", exist_ok=True)
logfile = f"logs/{uuid.uuid4()}.txt"
print(f"Logging to: {logfile}")

def print0(s, console=False, log=True):
    if console:
        print(s)
    if log:
        with open(logfile, "a") as f:
            print(s, file=f)

print0("="*100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
       + (f" on {torch.cuda.get_device_name(device)}" if device.type == "cuda" else " on CPU"), console=True)
print0("="*100)

val_tokens = 20 * 524288
batch_size = 8 * 64 * 1024
mbs = 8
val_inputs, val_targets = next(data_generator("finewebedu10B/finewebedu_val_*.bin", val_tokens))

REG_MODE = 'baseline'
SKETCH_DIM = 64
SIGR_ALPHA = 0.0

atma_config = AtmaConfig(vocab_size=50304, num_hidden_layers=12, hidden_size=768)
model = Model(atma_config, reg_mode=REG_MODE, sketch_dim=SKETCH_DIM).to(device)
model = torch.compile(model, dynamic=False, fullgraph=True)

print0(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", console=True)

# ----------------- MFU Setup Variables -----------------
num_params = sum(p.numel() for p in model.parameters())
seq_len = 1024
# Standard formulation (6 * N + 12 * L * H * Q * T) derived from PaLM Appendix B
flops_per_token = 6 * num_params + 12 * atma_config.num_hidden_layers * atma_config.hidden_size * seq_len
flops_per_step = flops_per_token * batch_size

peak_flops = 65e12 # default T4 fallback
if device.type == "cuda":
    gpu_name = torch.cuda.get_device_name()
    if "A100" in gpu_name:
        peak_flops = 312e12
    elif "V100" in gpu_name:
        peak_flops = 125e12
    elif "T4" in gpu_name:
        peak_flops = 65e12
    elif "L4" in gpu_name:
        peak_flops = 121e12
    elif "H100" in gpu_name:
        peak_flops = 989e12
    elif "H200" in gpu_name:
        peak_flops = 989e12

num_trials = 1

for _ in range(num_trials):

    ########################################
    #       Init & Optim Hyperparams       #
    ########################################

    # calculate total train steps based on available data
    train_files = sorted(Path.cwd().glob("finewebedu10B/finewebedu_train_*.bin"))
    train_steps = 0
    total_train_tokens = 0
    for f in train_files:
        header = torch.from_file(str(f), False, 256, dtype=torch.int32)
        num_tokens = int(header[2])
        total_train_tokens += num_tokens
        # Number of steps per shard mathematically aligns w/ generator (pos + batch_size + 1 >= num_tokens) condition
        train_steps += (num_tokens - 2) // batch_size

    print0(f"Calculated train_steps = {train_steps} from {total_train_tokens} tokens", console=True)

    # initialize model parameters
    for name, p in model.named_parameters():
        if "proj" in name:
            p.data.zero_()

    # create the optimizer(s)
    optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                        dict(params=[model.proj.weight], lr=1/320),
                        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
                      betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)
    optimizer2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2],
                  lr=0.02, weight_decay=0.01)
    
    optimizers = [optimizer1, optimizer2]

    assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())

    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # learning rate schedule: stable then decay
    def set_hparams(step, cooldown_frac=0.7):
        progress = step / train_steps
        assert 0 <= progress < 1
        if progress < 1 - cooldown_frac:
            eta = 1.0
        else:
            eta = (1 - progress) / cooldown_frac
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * eta

    ########################################
    #        Training and Validation       #
    ########################################

    train_loader = data_generator("finewebedu10B/finewebedu_train_*.bin", batch_size)

    # start the clock
    training_time = 0
    last_val_step = 0
    t0 = time.perf_counter()
    for step in range(train_steps + 1):

        # --------------- VALIDATION SECTION -----------------
        val_step_freq = 125 if step / train_steps < 0.9 else 25
        if step == train_steps or step % val_step_freq == 0:
            # stop the clock
            time_since_last_val = time.perf_counter() - t0
            step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float("nan")
            last_val_step = step
            training_time += time_since_last_val
            model.eval()
            val_loss = 0
            with torch.no_grad():
                assert len(val_inputs) % mbs == 0
                for i in range(len(val_inputs) // mbs):
                    val_loss_step, _ = model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
                    val_loss += val_loss_step.item()
            val_loss /= val_tokens

            mfu_str = ""
            if step > 0 and device.type == "cuda" and step_avg > 0:
                achieved_flops = flops_per_step / step_avg
                mfu = achieved_flops / peak_flops * 100
                mfu_str = f" MFU:{mfu:.1f}%"

            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
                   + f" step_avg:{1000*step_avg:.2f}ms{mfu_str}", console=True)
            model.train()
            # start the clock again
            t0 = time.perf_counter()

        if step == train_steps:
            break

        # --------------- TRAINING SECTION -----------------
        inputs, targets = next(train_loader)
        train_loss = 0

        # Microbatch (mbs) gradient accumulation runs successfully over batches
        assert len(inputs) % mbs == 0
        for i in range(len(inputs) // mbs):
            loss_step, reg_loss = model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
            train_loss += loss_step.item()

            loss = (1 - SIGR_ALPHA) * loss_step + (SIGR_ALPHA * reg_loss)
            loss.backward()

        train_loss /= batch_size

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # set optimization hyperparameters and take a step
        set_hparams(step)
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        approx_training_time = training_time + (time.perf_counter() - t0)
        curr_step_avg = approx_training_time / (step + 1)

        mfu_str_train = ""
        if device.type == "cuda" and curr_step_avg > 0:
            achieved_flops = flops_per_step / curr_step_avg
            mfu = achieved_flops / peak_flops * 100
            mfu_str_train = f" MFU:{mfu:.1f}%"

        print0(f"step:{step+1}/{train_steps} train_loss:{train_loss:.3f} train_time:{approx_training_time:.3f}s"
               + f" step_avg:{1000*curr_step_avg:.2f}ms{mfu_str_train}", console=True, log=False)