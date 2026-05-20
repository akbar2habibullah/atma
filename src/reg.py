import torch
import torch.nn.functional as F

def sigreg_weak_loss(x, sketch_dim=64):
    N, C = x.size()
    if C > sketch_dim:
        S = torch.randn(sketch_dim, C, device=x.device, dtype=x.dtype) / (C ** 0.5)
        x = x @ S.T
    else:
        sketch_dim = C
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / (N - 1 + 1e-6)
    target = torch.eye(sketch_dim, device=x.device)
    loss = torch.norm(cov - target, p='fro')
    return loss

def zipf_orthogonal_est(x, sketch_dim=64, zipf_s=1.0, lam_ang=1.0, lam_mag=1.0, eps=1e-6):
    N, C = x.size()
    if C > sketch_dim:
        S = torch.randn(sketch_dim, C, device=x.device, dtype=x.dtype) / (C ** 0.5)
        x = x @ S.T
        C = sketch_dim
    x = x - x.mean(dim=0, keepdim=True)
    norms = x.norm(dim=1, keepdim=True).clamp_min(eps)
    u = x / norms
    G = (u.T @ u) / (N - 1 + eps)
    ang_loss = torch.norm(G - torch.diag(torch.diag(G)), p='fro')
    sorted_norms, _ = torch.sort(norms.squeeze(-1), descending=True)
    ranks = torch.arange(1, N + 1, device=x.device, dtype=x.dtype)
    zipf_target = ranks.pow(-zipf_s)
    zipf_target = zipf_target / (zipf_target.sum() + eps)
    sorted_norms = sorted_norms / (sorted_norms.sum() + eps)
    mag_loss = torch.norm(sorted_norms - zipf_target, p='fro')
    return lam_ang * ang_loss + lam_mag * mag_loss

def sireg_discrete_loss(x, sketch_dim=64):
    N, C = x.size()
    x = F.normalize(x, p=2, dim=-1) * (C ** 0.5)
    if C > sketch_dim:
        S = torch.randn(sketch_dim, C, device=x.device, dtype=x.dtype) / (C ** 0.5)
        x = x @ S.T
    else:
        sketch_dim = C
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / (N - 1 + 1e-6)
    target = torch.eye(sketch_dim, device=x.device)
    loss = torch.norm(cov - target, p='fro')
    return loss

def sigreg_strong_loss(x, sketch_dim=64):
    N, C = x.size()
    A = torch.randn(C, sketch_dim, device=x.device, dtype=x.dtype)
    A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)
    t = torch.linspace(-5, 5, 17, device=x.device)
    exp_f = torch.exp(-0.5 * t**2)
    proj = x @ A
    args = proj.unsqueeze(2) * t.view(1, 1, -1)
    cos_mean = torch.cos(args).mean(dim=0)
    sin_mean = torch.sin(args).mean(dim=0)
    diff_sq = (cos_mean - exp_f.unsqueeze(0)).square() + sin_mean.square()
    err = diff_sq * exp_f.unsqueeze(0)
    loss = torch.trapz(err, t, dim=1) * N
    return loss.mean()

def sigreg(x, REG_MODE):
    reg_loss = torch.tensor(0.0, device=x.device)
    
    if REG_MODE != 'baseline':
        batch_size, seq_len, hidden_dim = x.shape
        flat_rep = x.reshape(-1, hidden_dim)
        if REG_MODE == 'weak':
            reg_loss = sigreg.sigreg_weak_loss(flat_rep, sketch_dim=64)
        elif REG_MODE == 'strong':
            reg_loss = sigreg.sigreg_strong_loss(flat_rep, sketch_dim=64)
        elif REG_MODE == 'discrete':
            reg_loss = sigreg.sireg_discrete_loss(flat_rep, sketch_dim=64)
        elif REG_MODE == 'zipfian':
            reg_loss = sigreg.zipf_orthogonal_est(flat_rep, sketch_dim=64)
    
    return reg_loss