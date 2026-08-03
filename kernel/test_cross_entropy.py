"""Parity tests for the soft-capped linear cross entropy head.

Run:
    python -m kernel.test_cross_entropy
"""

import torch

from kernel.cross_entropy import HAS_TRITON, softcap_linear_cross_entropy


torch.manual_seed(0)
PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  {extra}" if extra else ""))
    PASS += bool(cond)
    FAIL += not cond


def run_case(device, dtype, reduction, use_bias=True):
    bsz, seq, hidden, vocab = 3, 5, 17, 37
    x = torch.randn(bsz, seq, hidden, device=device, dtype=dtype, requires_grad=True)
    w = torch.randn(vocab, hidden, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(vocab, device=device, dtype=dtype, requires_grad=True) if use_bias else None
    y = torch.randint(0, vocab, (bsz, seq), device=device)
    y[0, 0] = -100

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    b_ref = None if bias is None else bias.detach().clone().requires_grad_(True)

    got = softcap_linear_cross_entropy(
        x,
        w,
        y,
        bias,
        reduction=reduction,
        impl="chunked",
        token_chunk_size=4,
        vocab_chunk_size=11,
    )
    ref = softcap_linear_cross_entropy(
        x_ref,
        w_ref,
        y,
        b_ref,
        reduction=reduction,
        impl="eager",
    )

    if reduction == "none":
        g = torch.randn_like(got)
        got.backward(g)
        ref.backward(g)
        loss_diff = (got - ref).abs().max().item()
    else:
        got.backward()
        ref.backward()
        loss_diff = abs(float(got) - float(ref))

    tol = 2e-5 if dtype == torch.float32 else 5e-2
    gx = (x.grad.float() - x_ref.grad.float()).abs().max().item()
    gw = (w.grad.float() - w_ref.grad.float()).abs().max().item()
    check(f"{device} {dtype} {reduction} loss", loss_diff < tol, f"diff={loss_diff:.2e}")
    check(f"{device} {dtype} {reduction} grad_x", gx < tol, f"diff={gx:.2e}")
    check(f"{device} {dtype} {reduction} grad_w", gw < tol, f"diff={gw:.2e}")
    if bias is not None:
        gb = (bias.grad.float() - b_ref.grad.float()).abs().max().item()
        check(f"{device} {dtype} {reduction} grad_b", gb < tol, f"diff={gb:.2e}")


for reduction in ("sum", "mean", "none"):
    run_case("cpu", torch.float32, reduction)

if torch.cuda.is_available():
    for dtype in (torch.float32, torch.bfloat16):
        for reduction in ("sum", "mean", "none"):
            run_case("cuda", dtype, reduction)

    if HAS_TRITON:
        x = torch.randn(2, 4, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        b = torch.randn(64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        y = torch.randint(0, 64, (2, 4), device="cuda")
        tri = softcap_linear_cross_entropy(x, w, y, b, impl="triton", vocab_chunk_size=32)
        ref = softcap_linear_cross_entropy(x, w, y, b, impl="chunked", vocab_chunk_size=17)
        check("cuda triton forward", abs(float(tri) - float(ref)) < 5e-2, f"diff={abs(float(tri)-float(ref)):.2e}")

print(f"\n{'=' * 50}\nResults: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
