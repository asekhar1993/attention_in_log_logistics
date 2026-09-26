# ----------------------------- model.py ------------------------------------
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from math import sqrt, pi, log

# ---------- helpers from your original code ----------
def inverse_softplus(x):
    return torch.log(torch.exp(x) - 1 + 1e-6)

def safe_inverse_softplus(x):
    return torch.log(torch.exp(x) - 1 + 1e-6)

def inverse_softplus_grad(x):
    return torch.exp(x) / (torch.exp(x) - 1 + 1e-6)

def logsumexp(a, dim, b):
    a_max = torch.max(a, dim=dim, keepdims=True)[0]
    out = torch.log(torch.sum(b * torch.exp(a - a_max), dim=dim, keepdims=True) + 1e-6)
    out += a_max
    return out
# ---------------------------------------------------------------------------


# ----------------- generic two‑layer MLP (unchanged) -----------------------
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x): return self.net(x)
# ---------------------------------------------------------------------------


class RegisterMoEMDN(nn.Module):
    def __init__(self, backbone,
                 input_size=384, hidden_size=256,
                 E=2, K=50, param_share=('mu','sigma'),
                 dropout=0.1, **_ignored):
        super().__init__()
        # rest of your init …

        self.E, self.K = E, K
        self.param_share = param_share
        self.backbone = backbone                      # e.g. ViT, CLIP‑proj …

        # ---------- gating network ----------
        self.gate = MLP(input_size, hidden_size, E, dropout)

        # ---------- per‑expert MDN heads ----------
        self.heads = nn.ModuleList([
            MLP(input_size, hidden_size,
                K * (3 - len(param_share)), dropout) for _ in range(E)
        ])

        # ---------- shared μ / σ if requested ----------
        shared = {}
        if 'mu' in param_share:
            shared['mu'] = nn.Parameter(torch.linspace(0, 200, K)
                                        .unsqueeze(0))           # (1,K)
        if 'sigma' in param_share:
            shared['sigma'] = nn.Parameter(torch.ones(1, K))
        self.shared = nn.ParameterDict(shared)

        # extra linear layers to allow shared μ/σ to be learnable
        self.layer_mu    = nn.Linear(K, K, bias=False)
        self.layer_sigma = nn.Linear(K, K, bias=False)

    # ------------ utilities (unchanged) ------------
    def _jacobian(self, t):
        return torch.exp(t) / (torch.exp(t) - 1 + 1e-6)          # |dy/dt|

    # def _cdf_single(self, w, mu, sigma, t):
    #     """
    #     CDF for *one* MDN (batch,B,K). Returns shape (B,) or (B,T).
    #     """
    #     y = safe_inverse_softplus(t).unsqueeze(-1)               # (...,1)
    #     normal = torch.distributions.Normal(mu, sigma)
    #     F_y = normal.cdf(y.repeat_interleave(self.K, -1))
    #     return (w * F_y).sum(-1)                                 # (...)


    def _cdf_single(self, w, mu, sigma, t):
        """
        CDF for *one* MDN (batch,B,K). Returns shape (B,) or (B,T).
        """
        # Check for NaN values in mu or sigma
        if torch.any(torch.isnan(mu)) or torch.any(torch.isnan(sigma)):
            print("NaN detected in mu or sigma!")
            # Replace NaN values with default values (e.g., 0 for mu, small value for sigma)
            mu = torch.nan_to_num(mu, nan=0.0)
            sigma = torch.nan_to_num(sigma, nan=1e-3)  # Prevent division by zero

        y = safe_inverse_softplus(t).unsqueeze(-1)  # (...,1)
        normal = torch.distributions.Normal(mu, sigma)
        F_y = normal.cdf(y.repeat_interleave(self.K, -1))
        return (w * F_y).sum(-1)  # (...)


    def cdf(self, params, t):  # keep public API unchanged
        return self._cdf_single(params['w'], params['mu'],
                                params['sigma'], t)

    def log_prob(self, params, t):
        y = safe_inverse_softplus(t).unsqueeze(-1)
        w, mu, sigma = params['w'], params['mu'], params['sigma']
        norm = torch.distributions.Normal(mu, sigma)
        log_pdf_y = logsumexp(norm.log_prob(y.repeat_interleave(self.K, -1)),
                              dim=-1, b=w).squeeze(-1)
        return log_pdf_y + torch.log(self._jacobian(t))

    # ------------ forward pass ------------
    # def forward(self, x, **kw):
    #     h_back = self.backbone(x, **kw)
    #     if isinstance(h_back, dict):
    #         subloss = h_back.get('loss')
    #         h = h_back['feat']
    #     else:
    #         subloss, h = None, h_back

    #     B = h.size(0)

    #     # ---- gating weights ----
    #     G = self.gate(h).softmax(-1)             # (B,E)

    #     # ---- collect expert‑specific params ----
    #     expert_params = []
    #     for e in range(self.E):
    #         out = self.heads[e](h)               # (B, K*(3-|share|))
    #         offset = 0
    #         p = {}
    #         for name in ('w','mu','sigma'):
    #             if name in self.param_share:
    #                 p[name] = self.shared[name]  # (1,K)
    #             else:
    #                 p[name] = out[:, offset:offset+self.K]
    #                 offset += self.K
    #         p['w'] = p['w'].softmax(-1)
    #         if 'mu' in self.param_share:
    #             p['mu'] = self.layer_mu(p['mu'])
    #         if 'sigma' in self.param_share:
    #             p['sigma'] = self.layer_sigma(p['sigma'])
    #         p['mu']    = p['mu'].clamp(-200, 200)
    #         p['sigma'] = F.softplus(p['sigma']).clamp(1e-2, 1e2)
    #         expert_params.append(p)             # list length E

    #     # ---- MoE aggregation ----
    #     # concatenate along component‑axis then weight by G
    #     w_stack   = torch.stack([p['w']    for p in expert_params], dim=1) # (B,E,K)
    #     mu_stack  = torch.stack([p['mu']   for p in expert_params], dim=1) # (B,E,K)
    #     sig_stack = torch.stack([p['sigma']for p in expert_params], dim=1) # (B,E,K)

    #     # mixture weights: G_e * λ_e,k → shape (B,E,K)
    #     W = G.unsqueeze(-1) * w_stack

    #     params = {
    #         'w':     W.view(B, -1, self.K).sum(1),   # (B,K)  final λ_k
    #         'mu':    (W * mu_stack).sum(1) / W.sum(1).clamp_min(1e-6),  # (B,K)
    #         'sigma': (W * sig_stack).sum(1) / W.sum(1).clamp_min(1e-6)
    #     }

    #     # ========== Extra Losses ==========

    #     # (2) Diversity regularization on μ
    #     # Pairwise squared L2 norm between expert means
    #     mu_diff = 0.0
    #     for i in range(self.E):
    #         for j in range(i+1, self.E):
    #             diff = (mu_stack[:, i] - mu_stack[:, j])  # (B, K)
    #             mu_diff += (diff ** 2).sum(-1).mean()     # scalar

    #     L_div = mu_diff / (self.E * (self.E - 1) / 2)

    #     # (3) Entropy regularization on gating weights
    #     L_ent = -(G * (G + 1e-8).log()).sum(-1).mean()

    #     extra_losses = {
    #         'L_div': L_div,
    #         'L_ent': L_ent
    #     }

    #     return params, subloss, extra_losses

    def forward(self, x, **kw):
        h_back = self.backbone(x, **kw)
        if isinstance(h_back, dict):
            subloss = h_back.get('loss')
            h = h_back['feat']
        else:
            subloss, h = None, h_back

        B = h.size(0)

        # ---- gating weights ----
        G = self.gate(h).softmax(-1)  # (B,E)

        # ---- collect expert‑specific params ----
        expert_params = []
        for e in range(self.E):
            out = self.heads[e](h)  # (B, K*(3-|share|))
            offset = 0
            p = {}
            for name in ('w', 'mu', 'sigma'):
                if name in self.param_share:
                    p[name] = self.shared[name]  # (1,K)
                else:
                    p[name] = out[:, offset:offset + self.K]
                    offset += self.K
            p['w'] = p['w'].softmax(-1)
            if 'mu' in self.param_share:
                p['mu'] = self.layer_mu(p['mu'])
            if 'sigma' in self.param_share:
                p['sigma'] = self.layer_sigma(p['sigma'])
            p['mu'] = p['mu'].clamp(-200, 200)
            p['sigma'] = F.softplus(p['sigma']).clamp(1e-2, 1e2)

            # Check for NaN values in mu or sigma
            if torch.any(torch.isnan(p['mu'])) or torch.any(torch.isnan(p['sigma'])):
                print(f"NaN detected in mu or sigma for expert {e}. Replacing with default values.")
                p['mu'] = torch.nan_to_num(p['mu'], nan=0.0)  # Replace NaNs in mu with default value (0)
                p['sigma'] = torch.nan_to_num(p['sigma'], nan=1e-3)  # Replace NaNs in sigma with default value (1e-3)

            expert_params.append(p)  # list length E

        # ---- MoE aggregation ----
        w_stack = torch.stack([p['w'] for p in expert_params], dim=1)  # (B,E,K)
        mu_stack = torch.stack([p['mu'] for p in expert_params], dim=1)  # (B,E,K)
        sig_stack = torch.stack([p['sigma'] for p in expert_params], dim=1)  # (B,E,K)

        # mixture weights: G_e * λ_e,k → shape (B,E,K)
        W = G.unsqueeze(-1) * w_stack

        params = {
            'w': W.view(B, -1, self.K).sum(1),  # (B,K)  final λ_k
            'mu': (W * mu_stack).sum(1) / W.sum(1).clamp_min(1e-6),  # (B,K)
            'sigma': (W * sig_stack).sum(1) / W.sum(1).clamp_min(1e-6)
        }

        # ========== Extra Losses ==========

        # (2) Diversity regularization on μ
        # Pairwise squared L2 norm between expert means
        mu_diff = 0.0
        for i in range(self.E):
            for j in range(i + 1, self.E):
                diff = (mu_stack[:, i] - mu_stack[:, j])  # (B, K)
                mu_diff += (diff ** 2).sum(-1).mean()     # scalar

        L_div = mu_diff / (self.E * (self.E - 1) / 2)

        # (3) Entropy regularization on gating weights
        L_ent = -(G * (G + 1e-8).log()).sum(-1).mean()

        extra_losses = {
            'L_div': L_div,
            'L_ent': L_ent
        }

        return params, subloss, extra_losses



    # def train_step(self, x_dict, t, c, constant_dict):
    #     params, subloss, extras = self(**x_dict)
    #     surv = 1. - self.cdf(params, t)
    #     pdf  = torch.exp(self.log_prob(params, t))

    #     return {
    #         't': t,
    #         'c': c,
    #         'pdf': pdf,
    #         'survival_func': surv,
    #         'subloss': subloss,
    #         'L_div': extras['L_div'],
    #         'L_ent': extras['L_ent']
    #     }
    def train_step(self, x_dict, t, c, constant_dict):
        params, subloss, extras = self(**x_dict)
    
        # Check if any NaN values exist in the parameters before using them
        if torch.any(torch.isnan(params['mu'])) or torch.any(torch.isnan(params['sigma'])):
            print("NaN detected in mu or sigma during train_step.")
            params['mu'] = torch.nan_to_num(params['mu'], nan=0.0)
            params['sigma'] = torch.nan_to_num(params['sigma'], nan=1e-3)
        
        surv = 1. - self.cdf(params, t)
        pdf = torch.exp(self.log_prob(params, t))

        return {
            't': t,
            'c': c,
            'pdf': pdf,
            'survival_func': surv,
            'subloss': subloss,
            'L_div': extras['L_div'],
            'L_ent': extras['L_ent']
        }


  
    def eval_step(self, x_dict, t, c, constant_dict):
        with torch.no_grad():
            params, _, extras = self(**x_dict)
            B = t.size(0)
            device = t.device
            outputs = {"t": t, "c": c, "L_div": extras['L_div'], "L_ent": extras['L_ent']}

            raw_eval_t = constant_dict["eval_t"].to(device)
            if raw_eval_t.dim() == 1:
                eval_t = raw_eval_t.unsqueeze(0).repeat(B, 1)
            elif raw_eval_t.dim() == 2:
                eval_t = raw_eval_t.repeat(B, 1) if raw_eval_t.size(0) == 1 else raw_eval_t
            else:
                raise ValueError(f"eval_t must be 1‑D or 2‑D, got {raw_eval_t.shape}")

            cdf = self.cdf(params, eval_t)
            surv = 1.0 - cdf
            haz = -torch.log(surv + 1e-8)
            outputs["eval_t"] = constant_dict["eval_t"]
            outputs["cum_hazard_seqs"] = haz.transpose(0, 1).contiguous()

            t_min, t_max = constant_dict["t_min"], constant_dict["t_max"]
            num_int_steps = int(constant_dict["NUM_INT_STEPS"])
            grid = torch.linspace(t_min.item(), t_max.item(), num_int_steps, dtype=t_min.dtype, device=device)
            surv_grid = 1.0 - self.cdf(params, grid.unsqueeze(0).repeat(B, 1))
            outputs["survival_seqs"] = surv_grid.transpose(0, 1).contiguous()

            for eps in (0.1, 0.2, 0.3, 0.4, 0.5):
                key = f"t_max_{eps}"
                if key not in constant_dict:
                    continue
                t_max_eps = constant_dict[key]
                grid_eps = torch.linspace(t_min.item(), t_max_eps.item(), num_int_steps, dtype=t_min.dtype, device=device)
                surv_eps = 1.0 - self.cdf(params, grid_eps.unsqueeze(0).repeat(B, 1))
                outputs[f"survival_seqs_{eps}"] = surv_eps.transpose(0, 1).contiguous()

            return outputs

    

    # def predict_step(self, x_dict):
    #     device = x_dict['x'].device
    #     with torch.no_grad():
    #         params, _ = self(**x_dict)
    #         t = torch.arange(0.1, 220.1, 0.1, device=device)
    #         surv = 1. - self.cdf(params, t)
    #     return {'t': t, 'p_survival': surv}


    def predict_step(self, x_dict):
        device = x_dict['x'].device
        with torch.no_grad():
            params, _, _ = self(**x_dict)  # Unpack three values, ignore subloss and extra_losses
            t = torch.arange(0.1, 220.1, 0.1, device=device)
            surv = 1. - self.cdf(params, t)
        return {'t': t, 'p_survival': surv}