# ############################################model log logistics in ISBI###############################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .AMIL import AMIL_layer

# def logsumexp(a, dim, b):
#     a_max = torch.max(a, dim=dim, keepdims=True)[0]
#     out = torch.log(torch.sum(b * torch.exp(a - a_max), dim=dim, keepdims=True) + 1e-12)
#     out = out + a_max
#     return out


# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, dropout=0.25):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_size, hidden_size), nn.GELU(), nn.Dropout(dropout),
#             nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout),
#             nn.Linear(hidden_size, output_size)
#         )

#     def forward(self, x):
#         return self.net(x)


# class EGMLL(nn.Module):
#     def __init__(self, backbone,
#                  input_size=384, hidden_size=256,
#                  E=2, K=50,
#                  dropout=0.1, **_ignored):
#         """
#         No parameter sharing for alpha and beta.
#         Each expert has its own alpha, beta heads.
#         """
#         super().__init__()
#         self.E, self.K = E, K
#         self.backbone = backbone

#         # # Gating network
#         # self.gate = MLP(input_size, hidden_size, E, dropout)

#         # # Each expert outputs (w, alpha, beta)
#         # head_dim = K * 3  # weights + alpha + beta
#         # Gating network
#         self.gate = MLP(input_size, hidden_size, E, dropout)
        
#         # --- NEW CHANGE ---
#         # Create E separate Attention modules (Attn_1 to Attn_e)
#         self.attentions = nn.ModuleList([
#             AMIL_layer(L=input_size, D=256, dropout=dropout) 
#             for _ in range(E)
#         ])
#         # -----------------

#         # Each expert outputs (w, alpha, beta)
#         head_dim = K * 3  # weights + alpha + beta
        
#         self.heads = nn.ModuleList([MLP(input_size, hidden_size, head_dim, dropout)
#                                     for _ in range(E)])

#         # Debug
#         self._debug_printed = False
#         self._debug_count = 0

#     # --- PDF / CDF utilities ---
#     def _get_pdf_cdf(self, alpha, beta, t):
#         if torch.any(torch.isnan(alpha)) or torch.any(torch.isnan(beta)):
#             alpha = torch.nan_to_num(alpha, nan=2.0)
#             beta = torch.nan_to_num(beta, nan=1.0)

#         alpha_pos = F.softplus(alpha).clamp(min=1e-3, max=50.0)
#         beta_pos = F.softplus(beta).clamp(min=1e-3, max=1e6)

#         if t.dim() == 1:
#             t_pos = t.clamp(min=1e-6).unsqueeze(-1)
#             log_ratio = (torch.log(t_pos) - torch.log(beta_pos)).clamp(-700, 700)
#             z = (alpha_pos * log_ratio).clamp(-700, 700)
#             cdf = torch.sigmoid(z)
#             log_denom = 2.0 * torch.log1p(torch.exp(z))
#             log_pdf = (torch.log(alpha_pos.clamp(min=1e-12)) -
#                        torch.log(beta_pos.clamp(min=1e-12)) +
#                        (alpha_pos - 1.0) * log_ratio -
#                        log_denom)
#             return log_pdf, cdf

#         elif t.dim() == 2:
#             t_pos = t.clamp(min=1e-6).unsqueeze(-1)
#             beta_exp = beta_pos.unsqueeze(1)
#             if t_pos.shape[0] != beta_exp.shape[0]:
#                 raise RuntimeError(f"_get_pdf_cdf batch mismatch: t_pos {t_pos.shape} vs beta {beta_exp.shape}")
#             log_ratio = (torch.log(t_pos) - torch.log(beta_exp)).clamp(-700, 700)
#             z = (alpha_pos.unsqueeze(1) * log_ratio).clamp(-700, 700)
#             cdf = torch.sigmoid(z)
#             log_denom = 2.0 * torch.log1p(torch.exp(z))
#             log_pdf = (torch.log(alpha_pos.clamp(min=1e-12)).unsqueeze(1) -
#                        torch.log(beta_pos.clamp(min=1e-12)).unsqueeze(1) +
#                        (alpha_pos.unsqueeze(1) - 1.0) * log_ratio -
#                        log_denom)
#             return log_pdf, cdf
#         else:
#             raise ValueError(f"_get_pdf_cdf: unexpected t.dim()={t.dim()}")

#     def _cdf_single(self, w, alpha, beta, t):
#         if t.dim() == 1:
#             _, cdf = self._get_pdf_cdf(alpha, beta, t)
#             return (w * cdf).sum(-1)
#         elif t.dim() == 2:
#             _, cdf = self._get_pdf_cdf(alpha, beta, t)
#             w_exp = w.unsqueeze(1)
#             return (w_exp * cdf).sum(-1)
#         else:
#             raise ValueError(f"_cdf_single: unexpected t.dim()={t.dim()}")

#     def cdf(self, params, t):
#         return self._cdf_single(params['w'], params['alpha'], params['beta'], t)

#     def log_prob(self, params, t):
#         log_pdf, _ = self._get_pdf_cdf(params['alpha'], params['beta'], t)
#         w = params['w']
#         if t.dim() == 1:
#             return logsumexp(log_pdf, dim=-1, b=w).squeeze(-1).clamp(min=-1e3, max=1e3)
#         else:
#             b = w.unsqueeze(1)
#             return logsumexp(log_pdf, dim=-1, b=b).squeeze(-1).clamp(min=-1e3, max=1e3)

#     def forward(self, x, **kw):
#         h_back = self.backbone(x, **kw)
#         if isinstance(h_back, dict):
#             subloss = h_back.get('loss')
#             h = h_back['feat']
#         else:
#             subloss, h = None, h_back

#         B = h.size(0)
#         if B == 0 or not torch.isfinite(h).all():
#             device = h.device
#             safe = {
#                 'w': torch.ones(B, self.K, device=device) / self.K,
#                 'alpha': torch.ones(B, self.K, device=device) * 2.0,
#                 'beta': torch.ones(B, self.K, device=device) * 10.0
#             }
#             return safe, subloss, {'L_ent': torch.tensor(0.0, device=device)}

#         # G = self.gate(h).softmax(-1)

#         # expert_params = []
#         # for e in range(self.E):
#         #     out = self.heads[e](h)
#         # --- NEW CHANGE ---
#         # 1. Gating network takes the mean of the patch features: G(mean(Pfinal))
#         G = self.gate(h.mean(dim=1)).softmax(-1)

#         expert_params = []
#         for e in range(self.E):
#             # 2. Each expert applies its own attention to Pfinal to get z^(e)
#             z_e, _, _ = self.attentions[e](h)
            
#             # 3. The expert head makes its prediction based on its unique z^(e)
#             out = self.heads[e](z_e)
#         # -----------------
#             if not torch.isfinite(out).all():
#                 out = torch.zeros_like(out)

#             offset = 0
#             p = {}
#             p['w'] = F.softmax(out[:, offset:offset + self.K], dim=-1)
#             offset += self.K
#             p['alpha'] = out[:, offset:offset + self.K]
#             offset += self.K
#             p['beta'] = out[:, offset:offset + self.K]

#             # Clamp via softplus
#             p['alpha'] = F.softplus(p['alpha']).clamp(0.1, 10.0)
#             p['beta'] = F.softplus(p['beta']).clamp(1.0, 200.0)

#             expert_params.append(p)

#         w_stack = torch.stack([p['w'] for p in expert_params], dim=1)
#         alpha_stack = torch.stack([p['alpha'] for p in expert_params], dim=1)
#         beta_stack = torch.stack([p['beta'] for p in expert_params], dim=1)

#         G_expanded = G.unsqueeze(-1)
#         w_weighted = torch.sum(G_expanded * w_stack, dim=1)
#         params_w = F.softmax(w_weighted, dim=-1)
#         alpha_weighted = torch.sum(G_expanded * alpha_stack, dim=1)
#         beta_weighted = torch.sum(G_expanded * beta_stack, dim=1)

#         if self.training:
#             noise_scale = 0.02
#             alpha_weighted = alpha_weighted + torch.randn_like(alpha_weighted) * noise_scale
#             beta_weighted = beta_weighted + torch.randn_like(beta_weighted) * (noise_scale * 5.0)

#         params = {
#             'w': params_w,
#             'alpha': F.softplus(alpha_weighted).clamp(0.1, 10.0),
#             'beta': F.softplus(beta_weighted).clamp(1.0, 200.0)
#         }

#         L_ent = -(G * (G + 1e-8).log()).sum(-1).mean()
#         extra_losses = {'L_ent': L_ent * 0.001}

#         self._debug_count += 1
#         if self._debug_count % 50 == 0:
#             print(f"DEBUG Forward #{self._debug_count} - alpha std {params['alpha'].std():.4f}, beta std {params['beta'].std():.4f}, gate std {G.std():.4f}")
#         return params, subloss, extra_losses

#     def train_step(self, x_dict, t, c, constant_dict):
#         params, subloss, extras = self(**x_dict)
#         if torch.any(torch.isnan(params['alpha'])) or torch.any(torch.isnan(params['beta'])) or torch.any(torch.isnan(params['w'])):
#             params['alpha'] = torch.ones_like(params['alpha']) * 2.0
#             params['beta'] = torch.ones_like(params['beta']) * 10.0
#             params['w'] = torch.ones_like(params['w']) / self.K

#         surv = 1.0 - self.cdf(params, t)
#         pdf_log = self.log_prob(params, t)
#         pdf = torch.exp(pdf_log).clamp(min=1e-8, max=1e6)
#         surv = surv.clamp(min=1e-8, max=1.0)

#         return {
#             't': t,
#             'c': c,
#             'pdf': pdf,
#             'survival_func': surv,
#             'subloss': subloss,
#             'L_ent': extras['L_ent']
#         }

#     def eval_step(self, x_dict, t, c, constant_dict):
#         with torch.no_grad():
#             params, _, extras = self(**x_dict)
#             B = t.size(0)
#             device = t.device
#             outputs = {"t": t, "c": c, "L_ent": extras['L_ent']}

#             raw_eval_t = constant_dict["eval_t"].to(device)
#             if raw_eval_t.dim() == 2 and raw_eval_t.size(0) == B:
#                 eval_t = raw_eval_t
#             else:
#                 t_vec = raw_eval_t.reshape(-1)
#                 eval_t = t_vec.unsqueeze(0).expand(B, t_vec.numel()).contiguous()

#             eval_t = eval_t.clamp(min=1e-6)

#             median_t = torch.median(eval_t, dim=-1, keepdim=True)[0]
#             cdf_median = self.cdf(params, median_t)
#             surv_median = 1.0 - cdf_median
#             risk_scores = -torch.log(surv_median.clamp(min=1e-12))

#             cdf = self.cdf(params, eval_t)
#             surv = 1.0 - cdf
#             haz = -torch.log(surv.clamp(min=1e-12))

#             haz_fixed = haz.clone()
#             haz_fixed[:, 0] = risk_scores.squeeze(-1)
#             outputs["cum_hazard_seqs"] = haz_fixed.transpose(0, 1).contiguous()

#             t_min = constant_dict["t_min"].to(device)
#             t_max = constant_dict["t_max"].to(device)
#             num_int_steps = int(constant_dict["NUM_INT_STEPS"])
#             grid = torch.linspace(float(t_min), float(t_max), num_int_steps, device=device)
#             surv_grid = 1.0 - self.cdf(params, grid.unsqueeze(0).repeat(B, 1))
#             outputs["survival_seqs"] = surv_grid.transpose(0, 1).contiguous()

#             for eps in (0.1, 0.2, 0.3, 0.4, 0.5):
#                 key = f"t_max_{eps}"
#                 if key in constant_dict:
#                     t_max_eps = constant_dict[key].to(device)
#                     grid_eps = torch.linspace(float(t_min), float(t_max_eps), num_int_steps, device=device)
#                     surv_eps = 1.0 - self.cdf(params, grid_eps.unsqueeze(0).repeat(B, 1))
#                     outputs[f"survival_seqs_{eps}"] = surv_eps.transpose(0, 1).contiguous()

#             outputs["eval_t"] = eval_t[0].detach().cpu()

#             if not self._debug_printed:
#                 self._debug_printed = True
#                 print(f"DEBUG eval_step: median_t={median_t[0,0]:.4f}, risk min {risk_scores.min():.4f}, max {risk_scores.max():.4f}")

#             return outputs

#     def predict_step(self, x_dict):
#         device = x_dict['x'].device
#         with torch.no_grad():
#             params, _, _ = self(**x_dict)
#             t = torch.arange(0.1, 220.1, 0.1, device=device).unsqueeze(0)
#             surv = 1.0 - self.cdf(params, t)
#         return {'t': t.squeeze(0), 'p_survival': surv.squeeze(0)}


##############################################################################for amil in egmll 1#########################################################

# import itertools

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from .AMIL import AMIL_layer


# def logsumexp(a, dim, b):
#     a_max = torch.max(a, dim=dim, keepdims=True)[0]
#     out = torch.log(torch.sum(b * torch.exp(a - a_max), dim=dim, keepdims=True) + 1e-12)
#     out = out + a_max
#     return out


# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, dropout=0.25):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_size, hidden_size), nn.GELU(), nn.Dropout(dropout),
#             nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout),
#             nn.Linear(hidden_size, output_size)
#         )

#     def forward(self, x):
#         return self.net(x)


# class EGMLL(nn.Module):
#     def __init__(self, backbone,
#                  input_size=384, hidden_size=256,
#                  E=5, K=50,
#                  dropout=0.1, diversity_weight=0.01, **_ignored):
#         """
#         Each expert now owns its own AMIL pooling head, so it forms its own bag
#         embedding (and its own attention distribution over patches) from the shared
#         patch-level backbone features, instead of all experts sharing one pooled
#         vector. The gate is decoupled from every expert's pooling: it only sees a
#         cheap mean-pooled summary, so it isn't biased toward whichever expert's
#         attention it might otherwise borrow.
#         """
#         super().__init__()
#         self.E, self.K = E, K
#         self.backbone = backbone
#         self.diversity_weight = diversity_weight

#         # Gating network: operates on a holistic, expert-independent slide summary.
#         self.gate = MLP(input_size, hidden_size, E, dropout)

#         # Each expert pools the shared patch features its own way.
#         self.expert_amils = nn.ModuleList([AMIL_layer(input_size, 256, dropout=dropout) for _ in range(E)])

#         # Each expert outputs (w, alpha, beta) from its own pooled embedding.
#         head_dim = K * 3
#         self.heads = nn.ModuleList([MLP(input_size, hidden_size, head_dim, dropout) for _ in range(E)])

#         # Debug
#         self._debug_printed = False
#         self._debug_count = 0

#     # --- PDF / CDF utilities (unchanged) ---
#     def _get_pdf_cdf(self, alpha, beta, t):
#         if torch.any(torch.isnan(alpha)) or torch.any(torch.isnan(beta)):
#             alpha = torch.nan_to_num(alpha, nan=2.0)
#             beta = torch.nan_to_num(beta, nan=1.0)

#         alpha_pos = F.softplus(alpha).clamp(min=1e-3, max=50.0)
#         beta_pos = F.softplus(beta).clamp(min=1e-3, max=1e6)

#         if t.dim() == 1:
#             t_pos = t.clamp(min=1e-6).unsqueeze(-1)
#             log_ratio = (torch.log(t_pos) - torch.log(beta_pos)).clamp(-700, 700)
#             z = (alpha_pos * log_ratio).clamp(-700, 700)
#             cdf = torch.sigmoid(z)
#             log_denom = 2.0 * torch.log1p(torch.exp(z))
#             log_pdf = (torch.log(alpha_pos.clamp(min=1e-12)) -
#                        torch.log(beta_pos.clamp(min=1e-12)) +
#                        (alpha_pos - 1.0) * log_ratio -
#                        log_denom)
#             return log_pdf, cdf

#         elif t.dim() == 2:
#             t_pos = t.clamp(min=1e-6).unsqueeze(-1)
#             beta_exp = beta_pos.unsqueeze(1)
#             if t_pos.shape[0] != beta_exp.shape[0]:
#                 raise RuntimeError(f"_get_pdf_cdf batch mismatch: t_pos {t_pos.shape} vs beta {beta_exp.shape}")
#             log_ratio = (torch.log(t_pos) - torch.log(beta_exp)).clamp(-700, 700)
#             z = (alpha_pos.unsqueeze(1) * log_ratio).clamp(-700, 700)
#             cdf = torch.sigmoid(z)
#             log_denom = 2.0 * torch.log1p(torch.exp(z))
#             log_pdf = (torch.log(alpha_pos.clamp(min=1e-12)).unsqueeze(1) -
#                        torch.log(beta_pos.clamp(min=1e-12)).unsqueeze(1) +
#                        (alpha_pos.unsqueeze(1) - 1.0) * log_ratio -
#                        log_denom)
#             return log_pdf, cdf
#         else:
#             raise ValueError(f"_get_pdf_cdf: unexpected t.dim()={t.dim()}")

#     def _cdf_single(self, w, alpha, beta, t):
#         if t.dim() == 1:
#             _, cdf = self._get_pdf_cdf(alpha, beta, t)
#             return (w * cdf).sum(-1)
#         elif t.dim() == 2:
#             _, cdf = self._get_pdf_cdf(alpha, beta, t)
#             w_exp = w.unsqueeze(1)
#             return (w_exp * cdf).sum(-1)
#         else:
#             raise ValueError(f"_cdf_single: unexpected t.dim()={t.dim()}")

#     def cdf(self, params, t):
#         return self._cdf_single(params['w'], params['alpha'], params['beta'], t)

#     def log_prob(self, params, t):
#         log_pdf, _ = self._get_pdf_cdf(params['alpha'], params['beta'], t)
#         w = params['w']
#         if t.dim() == 1:
#             return logsumexp(log_pdf, dim=-1, b=w).squeeze(-1).clamp(min=-1e3, max=1e3)
#         else:
#             b = w.unsqueeze(1)
#             return logsumexp(log_pdf, dim=-1, b=b).squeeze(-1).clamp(min=-1e3, max=1e3)

#     def _expert_forward(self, patch_feats, e):
#         """Run expert e's own AMIL pooling + head on the shared patch features."""
#         h_e, att_e, attn_logits_e = self.expert_amils[e](patch_feats)
#         out = self.heads[e](h_e)
#         if not torch.isfinite(out).all():
#             out = torch.zeros_like(out)

#         offset = 0
#         w = F.softmax(out[:, offset:offset + self.K], dim=-1)
#         offset += self.K
#         alpha = out[:, offset:offset + self.K]
#         offset += self.K
#         beta = out[:, offset:offset + self.K]

#         alpha = F.softplus(alpha).clamp(0.1, 10.0)
#         beta = F.softplus(beta).clamp(1.0, 200.0)

#         return {'w': w, 'alpha': alpha, 'beta': beta}, att_e, h_e

#     def forward(self, x, return_interpret=False, **kw):
#         h_back = self.backbone(x, **kw)
#         if isinstance(h_back, dict):
#             subloss = h_back.get('loss')
#             patch_feats = h_back['feat']
#             patch_order = h_back.get('patch_order')
#         else:
#             subloss, patch_feats, patch_order = None, h_back, None

#         B = patch_feats.size(0)
#         if B == 0 or not torch.isfinite(patch_feats).all():
#             device = patch_feats.device
#             L = max(patch_feats.size(1), 1)
#             patch_feats = torch.zeros(B, L, patch_feats.size(-1), device=device)

#         # Gate: a cheap, expert-independent holistic summary. Deliberately NOT
#         # derived from any single expert's AMIL output, so it doesn't inherit that
#         # expert's attention bias.
#         gate_summary = patch_feats.mean(dim=1)
#         G = self.gate(gate_summary).softmax(-1)

#         expert_params, expert_atts, expert_hs = [], [], []
#         for e in range(self.E):
#             p_e, att_e, h_e = self._expert_forward(patch_feats, e)
#             expert_params.append(p_e)
#             expert_atts.append(att_e)
#             expert_hs.append(h_e)

#         w_stack = torch.stack([p['w'] for p in expert_params], dim=1)
#         alpha_stack = torch.stack([p['alpha'] for p in expert_params], dim=1)
#         beta_stack = torch.stack([p['beta'] for p in expert_params], dim=1)

#         G_expanded = G.unsqueeze(-1)
#         w_weighted = torch.sum(G_expanded * w_stack, dim=1)
#         params_w = F.softmax(w_weighted, dim=-1)
#         alpha_weighted = torch.sum(G_expanded * alpha_stack, dim=1)
#         beta_weighted = torch.sum(G_expanded * beta_stack, dim=1)

#         if self.training:
#             noise_scale = 0.02
#             alpha_weighted = alpha_weighted + torch.randn_like(alpha_weighted) * noise_scale
#             beta_weighted = beta_weighted + torch.randn_like(beta_weighted) * (noise_scale * 5.0)

#         params = {
#             'w': params_w,
#             'alpha': F.softplus(alpha_weighted).clamp(0.1, 10.0),
#             'beta': F.softplus(beta_weighted).clamp(1.0, 200.0)
#         }

#         L_ent = -(G * (G + 1e-8).log()).sum(-1).mean()
#         extra_losses = {'L_ent': L_ent * 0.001}

#         # Diversity penalty: nothing else stops the E experts from converging to
#         # near-identical bag embeddings (and therefore near-identical attention).
#         # Penalize pairwise cosine similarity between expert bag embeddings.
#         if self.E > 1:
#             pairs = list(itertools.combinations(range(self.E), 2))
#             sims = torch.stack([
#                 F.cosine_similarity(expert_hs[i], expert_hs[j], dim=-1).mean()
#                 for i, j in pairs
#             ])
#             extra_losses['L_div'] = sims.mean() * self.diversity_weight

#         if return_interpret:
#             # Everything a separate plotting script needs, per forward call — no
#             # saving to disk here, that's left to that script.
#             extra_losses['expert_atts'] = expert_atts        # list of E tensors, each B x L
#             extra_losses['patch_order'] = patch_order        # B x L (or None), maps position -> original patch index
#             extra_losses['gate'] = G                         # B x E

#         self._debug_count += 1
#         if self._debug_count % 50 == 0:
#             print(f"DEBUG Forward #{self._debug_count} - alpha std {params['alpha'].std():.4f}, "
#                   f"beta std {params['beta'].std():.4f}, gate std {G.std():.4f}")
#         return params, subloss, extra_losses

#     def train_step(self, x_dict, t, c, constant_dict):
#         params, subloss, extras = self(**x_dict)
#         if torch.any(torch.isnan(params['alpha'])) or torch.any(torch.isnan(params['beta'])) or torch.any(torch.isnan(params['w'])):
#             params['alpha'] = torch.ones_like(params['alpha']) * 2.0
#             params['beta'] = torch.ones_like(params['beta']) * 10.0
#             params['w'] = torch.ones_like(params['w']) / self.K

#         surv = 1.0 - self.cdf(params, t)
#         pdf_log = self.log_prob(params, t)
#         pdf = torch.exp(pdf_log).clamp(min=1e-8, max=1e6)
#         surv = surv.clamp(min=1e-8, max=1.0)
#         return {
#             't': t,
#             'c': c,
#             'pdf': pdf,
#             'survival_func': surv,
#             'subloss': subloss,
#             'L_ent': extras['L_ent'],
#             'L_div': extras.get('L_div', torch.zeros((), device=t.device)),
#         }

#     def eval_step(self, x_dict, t, c, constant_dict):
#         with torch.no_grad():
#             params, _, extras = self(**x_dict)
#             B = t.size(0)
#             device = t.device
#             outputs = {"t": t, "c": c, "L_ent": extras['L_ent']}

#             raw_eval_t = constant_dict["eval_t"].to(device)
#             if raw_eval_t.dim() == 2 and raw_eval_t.size(0) == B:
#                 eval_t = raw_eval_t
#             else:
#                 t_vec = raw_eval_t.reshape(-1)
#                 eval_t = t_vec.unsqueeze(0).expand(B, t_vec.numel()).contiguous()

#             eval_t = eval_t.clamp(min=1e-6)

#             median_t = torch.median(eval_t, dim=-1, keepdim=True)[0]
#             cdf_median = self.cdf(params, median_t)
#             surv_median = 1.0 - cdf_median
#             risk_scores = -torch.log(surv_median.clamp(min=1e-12))

#             cdf = self.cdf(params, eval_t)
#             surv = 1.0 - cdf
#             haz = -torch.log(surv.clamp(min=1e-12))

#             haz_fixed = haz.clone()
#             haz_fixed[:, 0] = risk_scores.squeeze(-1)
#             outputs["cum_hazard_seqs"] = haz_fixed.transpose(0, 1).contiguous()

#             t_min = constant_dict["t_min"].to(device)
#             t_max = constant_dict["t_max"].to(device)
#             num_int_steps = int(constant_dict["NUM_INT_STEPS"])
#             grid = torch.linspace(float(t_min), float(t_max), num_int_steps, device=device)
#             surv_grid = 1.0 - self.cdf(params, grid.unsqueeze(0).repeat(B, 1))
#             outputs["survival_seqs"] = surv_grid.transpose(0, 1).contiguous()

#             for eps in (0.1, 0.2, 0.3, 0.4, 0.5):
#                 key = f"t_max_{eps}"
#                 if key in constant_dict:
#                     t_max_eps = constant_dict[key].to(device)
#                     grid_eps = torch.linspace(float(t_min), float(t_max_eps), num_int_steps, device=device)
#                     surv_eps = 1.0 - self.cdf(params, grid_eps.unsqueeze(0).repeat(B, 1))
#                     outputs[f"survival_seqs_{eps}"] = surv_eps.transpose(0, 1).contiguous()

#             outputs["eval_t"] = eval_t[0].detach().cpu()

#             if not self._debug_printed:
#                 self._debug_printed = True
#                 print(f"DEBUG eval_step: median_t={median_t[0,0]:.4f}, risk min {risk_scores.min():.4f}, max {risk_scores.max():.4f}")

#             return outputs

#     def predict_step(self, x_dict):
#         device = x_dict['x'].device
#         with torch.no_grad():
#             params, _, _ = self(**x_dict)
#             t = torch.arange(0.1, 220.1, 0.1, device=device).unsqueeze(0)
#             surv = 1.0 - self.cdf(params, t)
#         return {'t': t.squeeze(0), 'p_survival': surv.squeeze(0)}

############################################for amil in egmll 2###############################################################

# import itertools

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from .AMIL import AMIL_layer


# def logsumexp(a, dim, b):
#     a_max = torch.max(a, dim=dim, keepdims=True)[0]
#     out = torch.log(torch.sum(b * torch.exp(a - a_max), dim=dim, keepdims=True) + 1e-12)
#     out = out + a_max
#     return out


# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, dropout=0.25):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_size, hidden_size), nn.GELU(), nn.Dropout(dropout),
#             nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout),
#             nn.Linear(hidden_size, output_size)
#         )

#     def forward(self, x):
#         return self.net(x)


# class EGMLL(nn.Module):
#     def __init__(self, backbone,
#                  input_size=384, hidden_size=256,
#                  E=5, K=50,
#                  dropout_rate=0.25, diversity_weight=0.01, **_ignored):
#         """
#         Each expert now owns its own AMIL pooling head, so it forms its own bag
#         embedding (and its own attention distribution over patches) from the shared
#         patch-level backbone features, instead of all experts sharing one pooled
#         vector. The gate is decoupled from every expert's pooling: it only sees a
#         cheap mean-pooled summary, so it isn't biased toward whichever expert's
#         attention it might otherwise borrow.
#         """
#         super().__init__()
#         self.E, self.K = E, K
#         self.backbone = backbone
#         self.diversity_weight = diversity_weight

#         # Gating network: operates on a holistic, expert-independent slide summary.
#         self.gate = MLP(input_size, hidden_size, E, dropout_rate)

#         # Each expert pools the shared patch features its own way.
#         self.expert_amils = nn.ModuleList([AMIL_layer(input_size, 256, dropout=dropout_rate) for _ in range(E)])

#         # Each expert outputs (w, alpha, beta) from its own pooled embedding.
#         head_dim = K * 3
#         self.heads = nn.ModuleList([MLP(input_size, hidden_size, head_dim, dropout_rate) for _ in range(E)])

#         # Debug
#         self._debug_printed = False
#         self._debug_count = 0

#     # --- PDF / CDF utilities (unchanged) ---
#     def _get_pdf_cdf(self, alpha, beta, t):
#         if torch.any(torch.isnan(alpha)) or torch.any(torch.isnan(beta)):
#             alpha = torch.nan_to_num(alpha, nan=2.0)
#             beta = torch.nan_to_num(beta, nan=1.0)

#         alpha_pos = F.softplus(alpha).clamp(min=1e-3, max=50.0)
#         beta_pos = F.softplus(beta).clamp(min=1e-3, max=1e6)

#         if t.dim() == 1:
#             t_pos = t.clamp(min=1e-6).unsqueeze(-1)
#             log_ratio = (torch.log(t_pos) - torch.log(beta_pos)).clamp(-700, 700)
#             z = (alpha_pos * log_ratio).clamp(-700, 700)
#             cdf = torch.sigmoid(z)
#             log_denom = 2.0 * torch.log1p(torch.exp(z))
#             log_pdf = (torch.log(alpha_pos.clamp(min=1e-12)) -
#                        torch.log(beta_pos.clamp(min=1e-12)) +
#                        (alpha_pos - 1.0) * log_ratio -
#                        log_denom)
#             return log_pdf, cdf

#         elif t.dim() == 2:
#             t_pos = t.clamp(min=1e-6).unsqueeze(-1)
#             beta_exp = beta_pos.unsqueeze(1)
#             if t_pos.shape[0] != beta_exp.shape[0]:
#                 raise RuntimeError(f"_get_pdf_cdf batch mismatch: t_pos {t_pos.shape} vs beta {beta_exp.shape}")
#             log_ratio = (torch.log(t_pos) - torch.log(beta_exp)).clamp(-700, 700)
#             z = (alpha_pos.unsqueeze(1) * log_ratio).clamp(-700, 700)
#             cdf = torch.sigmoid(z)
#             log_denom = 2.0 * torch.log1p(torch.exp(z))
#             log_pdf = (torch.log(alpha_pos.clamp(min=1e-12)).unsqueeze(1) -
#                        torch.log(beta_pos.clamp(min=1e-12)).unsqueeze(1) +
#                        (alpha_pos.unsqueeze(1) - 1.0) * log_ratio -
#                        log_denom)
#             return log_pdf, cdf
#         else:
#             raise ValueError(f"_get_pdf_cdf: unexpected t.dim()={t.dim()}")

#     def _cdf_single(self, w, alpha, beta, t):
#         if t.dim() == 1:
#             _, cdf = self._get_pdf_cdf(alpha, beta, t)
#             return (w * cdf).sum(-1)
#         elif t.dim() == 2:
#             _, cdf = self._get_pdf_cdf(alpha, beta, t)
#             w_exp = w.unsqueeze(1)
#             return (w_exp * cdf).sum(-1)
#         else:
#             raise ValueError(f"_cdf_single: unexpected t.dim()={t.dim()}")

#     def cdf(self, params, t):
#         return self._cdf_single(params['w'], params['alpha'], params['beta'], t)

#     def log_prob(self, params, t):
#         log_pdf, _ = self._get_pdf_cdf(params['alpha'], params['beta'], t)
#         w = params['w']
#         if t.dim() == 1:
#             return logsumexp(log_pdf, dim=-1, b=w).squeeze(-1).clamp(min=-1e3, max=1e3)
#         else:
#             b = w.unsqueeze(1)
#             return logsumexp(log_pdf, dim=-1, b=b).squeeze(-1).clamp(min=-1e3, max=1e3)

#     def _expert_forward(self, patch_feats, e):
#         """Run expert e's own AMIL pooling + head on the shared patch features."""
#         h_e, att_e, attn_logits_e = self.expert_amils[e](patch_feats)
#         out = self.heads[e](h_e)
#         if not torch.isfinite(out).all():
#             out = torch.zeros_like(out)

#         offset = 0
#         w = F.softmax(out[:, offset:offset + self.K], dim=-1)
#         offset += self.K
#         alpha = out[:, offset:offset + self.K]
#         offset += self.K
#         beta = out[:, offset:offset + self.K]

#         alpha = F.softplus(alpha).clamp(0.1, 10.0)
#         beta = F.softplus(beta).clamp(1.0, 200.0)

#         return {'w': w, 'alpha': alpha, 'beta': beta}, att_e, h_e

#     def forward(self, x, return_interpret=False, **kw):
#         h_back = self.backbone(x, **kw)
#         if isinstance(h_back, dict):
#             subloss = h_back.get('loss')
#             patch_feats = h_back['feat']
#             patch_order = h_back.get('patch_order')
#         else:
#             subloss, patch_feats, patch_order = None, h_back, None

#         B = patch_feats.size(0)
#         if B == 0 or not torch.isfinite(patch_feats).all():
#             device = patch_feats.device
#             L = max(patch_feats.size(1), 1)
#             patch_feats = torch.zeros(B, L, patch_feats.size(-1), device=device)

#         # Gate: a cheap, expert-independent holistic summary. Deliberately NOT
#         # derived from any single expert's AMIL output, so it doesn't inherit that
#         # expert's attention bias.
#         gate_summary = patch_feats.mean(dim=1)
#         G = self.gate(gate_summary).softmax(-1)

#         expert_params, expert_atts, expert_hs = [], [], []
#         for e in range(self.E):
#             p_e, att_e, h_e = self._expert_forward(patch_feats, e)
#             expert_params.append(p_e)
#             expert_atts.append(att_e)
#             expert_hs.append(h_e)

#         w_stack = torch.stack([p['w'] for p in expert_params], dim=1)
#         alpha_stack = torch.stack([p['alpha'] for p in expert_params], dim=1)
#         beta_stack = torch.stack([p['beta'] for p in expert_params], dim=1)

#         G_expanded = G.unsqueeze(-1)
#         w_weighted = torch.sum(G_expanded * w_stack, dim=1)
#         params_w = F.softmax(w_weighted, dim=-1)
#         alpha_weighted = torch.sum(G_expanded * alpha_stack, dim=1)
#         beta_weighted = torch.sum(G_expanded * beta_stack, dim=1)

#         if self.training:
#             noise_scale = 0.02
#             alpha_weighted = alpha_weighted + torch.randn_like(alpha_weighted) * noise_scale
#             beta_weighted = beta_weighted + torch.randn_like(beta_weighted) * (noise_scale * 5.0)

#         params = {
#             'w': params_w,
#             'alpha': F.softplus(alpha_weighted).clamp(0.1, 10.0),
#             'beta': F.softplus(beta_weighted).clamp(1.0, 200.0)
#         }

#         L_ent = -(G * (G + 1e-8).log()).sum(-1).mean()
#         extra_losses = {'L_ent': L_ent * 0.001}

#         # Diversity penalty: nothing else stops the E experts from converging to
#         # near-identical bag embeddings (and therefore near-identical attention).
#         # Penalize pairwise cosine similarity between expert bag embeddings.
#         if self.E > 1:
#             pairs = list(itertools.combinations(range(self.E), 2))
#             sims = torch.stack([
#                 F.cosine_similarity(expert_hs[i], expert_hs[j], dim=-1).mean()
#                 for i, j in pairs
#             ])
#             extra_losses['L_div'] = sims.mean() * self.diversity_weight

#         if return_interpret:
#             # Everything a separate plotting script needs, per forward call — no
#             # saving to disk here, that's left to that script.
#             extra_losses['expert_atts'] = expert_atts        # list of E tensors, each B x L
#             extra_losses['patch_order'] = patch_order        # B x L (or None), maps position -> original patch index
#             extra_losses['gate'] = G                         # B x E

#         self._debug_count += 1
#         if self._debug_count % 50 == 0:
#             print(f"DEBUG Forward #{self._debug_count} - alpha std {params['alpha'].std():.4f}, "
#                   f"beta std {params['beta'].std():.4f}, gate std {G.std():.4f}")
#         return params, subloss, extra_losses

#     def train_step(self, x_dict, t, c, constant_dict):
#         params, subloss, extras = self(**x_dict)
#         if torch.any(torch.isnan(params['alpha'])) or torch.any(torch.isnan(params['beta'])) or torch.any(torch.isnan(params['w'])):
#             params['alpha'] = torch.ones_like(params['alpha']) * 2.0
#             params['beta'] = torch.ones_like(params['beta']) * 10.0
#             params['w'] = torch.ones_like(params['w']) / self.K

#         surv = 1.0 - self.cdf(params, t)
#         pdf_log = self.log_prob(params, t)
#         pdf = torch.exp(pdf_log).clamp(min=1e-8, max=1e6)
#         surv = surv.clamp(min=1e-8, max=1.0)
#         return {
#             't': t,
#             'c': c,
#             'pdf': pdf,
#             'survival_func': surv,
#             'subloss': subloss,
#             'L_ent': extras['L_ent'],
#             'L_div': extras.get('L_div', torch.zeros((), device=t.device)),
#         }

#     def eval_step(self, x_dict, t, c, constant_dict):
#         with torch.no_grad():
#             params, _, extras = self(**x_dict)
#             B = t.size(0)
#             device = t.device
#             outputs = {"t": t, "c": c, "L_ent": extras['L_ent']}

#             raw_eval_t = constant_dict["eval_t"].to(device)
#             if raw_eval_t.dim() == 2 and raw_eval_t.size(0) == B:
#                 eval_t = raw_eval_t
#             else:
#                 t_vec = raw_eval_t.reshape(-1)
#                 eval_t = t_vec.unsqueeze(0).expand(B, t_vec.numel()).contiguous()

#             eval_t = eval_t.clamp(min=1e-6)

#             median_t = torch.median(eval_t, dim=-1, keepdim=True)[0]
#             cdf_median = self.cdf(params, median_t)
#             surv_median = 1.0 - cdf_median
#             risk_scores = -torch.log(surv_median.clamp(min=1e-12))

#             cdf = self.cdf(params, eval_t)
#             surv = 1.0 - cdf
#             haz = -torch.log(surv.clamp(min=1e-12))

#             haz_fixed = haz.clone()
#             haz_fixed[:, 0] = risk_scores.squeeze(-1)
#             outputs["cum_hazard_seqs"] = haz_fixed.transpose(0, 1).contiguous()

#             t_min = constant_dict["t_min"].to(device)
#             t_max = constant_dict["t_max"].to(device)
#             num_int_steps = int(constant_dict["NUM_INT_STEPS"])
#             grid = torch.linspace(float(t_min), float(t_max), num_int_steps, device=device)
#             surv_grid = 1.0 - self.cdf(params, grid.unsqueeze(0).repeat(B, 1))
#             outputs["survival_seqs"] = surv_grid.transpose(0, 1).contiguous()

#             for eps in (0.1, 0.2, 0.3, 0.4, 0.5):
#                 key = f"t_max_{eps}"
#                 if key in constant_dict:
#                     t_max_eps = constant_dict[key].to(device)
#                     grid_eps = torch.linspace(float(t_min), float(t_max_eps), num_int_steps, device=device)
#                     surv_eps = 1.0 - self.cdf(params, grid_eps.unsqueeze(0).repeat(B, 1))
#                     outputs[f"survival_seqs_{eps}"] = surv_eps.transpose(0, 1).contiguous()

#             outputs["eval_t"] = eval_t[0].detach().cpu()

#             if not self._debug_printed:
#                 self._debug_printed = True
#                 print(f"DEBUG eval_step: median_t={median_t[0,0]:.4f}, risk min {risk_scores.min():.4f}, max {risk_scores.max():.4f}")

#             return outputs

#     def predict_step(self, x_dict):
#         device = x_dict['x'].device
#         with torch.no_grad():
#             params, _, _ = self(**x_dict)
#             t = torch.arange(0.1, 220.1, 0.1, device=device).unsqueeze(0)
#             surv = 1.0 - self.cdf(params, t)
#         return {'t': t.squeeze(0), 'p_survival': surv.squeeze(0)}

###########################################################for amil in egmll 3##########################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

from .AMIL import AMIL_layer


def logsumexp(a, dim, b):
    a_max = torch.max(a, dim=dim, keepdims=True)[0]
    out = torch.log(torch.sum(b * torch.exp(a - a_max), dim=dim, keepdims=True) + 1e-12)
    out = out + a_max
    return out


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)


class EGMLL(nn.Module):
    def __init__(self, backbone,
                 input_size=384, hidden_size=256,
                 E=5, K=50,
                 dropout_rate=0.25, **_ignored):
        """
        Each expert now owns its own AMIL pooling head, so it forms its own bag
        embedding (and its own attention distribution over patches) from the shared
        patch-level backbone features, instead of all experts sharing one pooled
        vector. The gate is decoupled from every expert's pooling: it only sees a
        cheap mean-pooled summary, so it isn't biased toward whichever expert's
        attention it might otherwise borrow.
        """
        super().__init__()
        self.E, self.K = E, K
        self.backbone = backbone

        # Gating network: operates on a holistic, expert-independent slide summary.
        self.gate = MLP(input_size, hidden_size, E, dropout_rate)

        # Each expert pools the shared patch features its own way.
        self.expert_amils = nn.ModuleList([AMIL_layer(input_size, 256, dropout=dropout_rate) for _ in range(E)])

        # Each expert outputs (w, alpha, beta) from its own pooled embedding.
        head_dim = K * 3
        self.heads = nn.ModuleList([MLP(input_size, hidden_size, head_dim, dropout_rate) for _ in range(E)])

        # Debug
        self._debug_printed = False
        self._debug_count = 0

    # --- PDF / CDF utilities (unchanged) ---
    def _get_pdf_cdf(self, alpha, beta, t):
        if torch.any(torch.isnan(alpha)) or torch.any(torch.isnan(beta)):
            alpha = torch.nan_to_num(alpha, nan=2.0)
            beta = torch.nan_to_num(beta, nan=1.0)

        alpha_pos = F.softplus(alpha).clamp(min=1e-3, max=50.0)
        beta_pos = F.softplus(beta).clamp(min=1e-3, max=1e6)

        if t.dim() == 1:
            t_pos = t.clamp(min=1e-6).unsqueeze(-1)
            log_ratio = (torch.log(t_pos) - torch.log(beta_pos)).clamp(-700, 700)
            z = (alpha_pos * log_ratio).clamp(-700, 700)
            cdf = torch.sigmoid(z)
            log_denom = 2.0 * torch.log1p(torch.exp(z))
            log_pdf = (torch.log(alpha_pos.clamp(min=1e-12)) -
                       torch.log(beta_pos.clamp(min=1e-12)) +
                       (alpha_pos - 1.0) * log_ratio -
                       log_denom)
            return log_pdf, cdf

        elif t.dim() == 2:
            t_pos = t.clamp(min=1e-6).unsqueeze(-1)
            beta_exp = beta_pos.unsqueeze(1)
            if t_pos.shape[0] != beta_exp.shape[0]:
                raise RuntimeError(f"_get_pdf_cdf batch mismatch: t_pos {t_pos.shape} vs beta {beta_exp.shape}")
            log_ratio = (torch.log(t_pos) - torch.log(beta_exp)).clamp(-700, 700)
            z = (alpha_pos.unsqueeze(1) * log_ratio).clamp(-700, 700)
            cdf = torch.sigmoid(z)
            log_denom = 2.0 * torch.log1p(torch.exp(z))
            log_pdf = (torch.log(alpha_pos.clamp(min=1e-12)).unsqueeze(1) -
                       torch.log(beta_pos.clamp(min=1e-12)).unsqueeze(1) +
                       (alpha_pos.unsqueeze(1) - 1.0) * log_ratio -
                       log_denom)
            return log_pdf, cdf
        else:
            raise ValueError(f"_get_pdf_cdf: unexpected t.dim()={t.dim()}")

    def _cdf_single(self, w, alpha, beta, t):
        if t.dim() == 1:
            _, cdf = self._get_pdf_cdf(alpha, beta, t)
            return (w * cdf).sum(-1)
        elif t.dim() == 2:
            _, cdf = self._get_pdf_cdf(alpha, beta, t)
            w_exp = w.unsqueeze(1)
            return (w_exp * cdf).sum(-1)
        else:
            raise ValueError(f"_cdf_single: unexpected t.dim()={t.dim()}")

    def cdf(self, params, t):
        return self._cdf_single(params['w'], params['alpha'], params['beta'], t)

    def log_prob(self, params, t):
        log_pdf, _ = self._get_pdf_cdf(params['alpha'], params['beta'], t)
        w = params['w']
        if t.dim() == 1:
            return logsumexp(log_pdf, dim=-1, b=w).squeeze(-1).clamp(min=-1e3, max=1e3)
        else:
            b = w.unsqueeze(1)
            return logsumexp(log_pdf, dim=-1, b=b).squeeze(-1).clamp(min=-1e3, max=1e3)

    def _expert_forward(self, patch_feats, e):
        """Run expert e's own AMIL pooling + head on the shared patch features."""
        h_e, att_e, attn_logits_e = self.expert_amils[e](patch_feats)
        out = self.heads[e](h_e)
        if not torch.isfinite(out).all():
            out = torch.zeros_like(out)

        offset = 0
        w = F.softmax(out[:, offset:offset + self.K], dim=-1)
        offset += self.K
        alpha = out[:, offset:offset + self.K]
        offset += self.K
        beta = out[:, offset:offset + self.K]

        alpha = F.softplus(alpha).clamp(0.1, 10.0)
        beta = F.softplus(beta).clamp(1.0, 200.0)

        return {'w': w, 'alpha': alpha, 'beta': beta}, att_e, h_e

    def forward(self, x, return_interpret=False, **kw):
        h_back = self.backbone(x, **kw)
        if isinstance(h_back, dict):
            subloss = h_back.get('loss')
            patch_feats = h_back['feat']
            patch_order = h_back.get('patch_order')
        else:
            subloss, patch_feats, patch_order = None, h_back, None

        B = patch_feats.size(0)
        if B == 0 or not torch.isfinite(patch_feats).all():
            device = patch_feats.device
            L = max(patch_feats.size(1), 1)
            patch_feats = torch.zeros(B, L, patch_feats.size(-1), device=device)

        # Gate: a cheap, expert-independent holistic summary. Deliberately NOT
        # derived from any single expert's AMIL output, so it doesn't inherit that
        # expert's attention bias.
        gate_summary = patch_feats.mean(dim=1)
        G = self.gate(gate_summary).softmax(-1)

        expert_params, expert_atts, expert_hs = [], [], []
        for e in range(self.E):
            p_e, att_e, h_e = self._expert_forward(patch_feats, e)
            expert_params.append(p_e)
            expert_atts.append(att_e)
            expert_hs.append(h_e)

        w_stack = torch.stack([p['w'] for p in expert_params], dim=1)
        alpha_stack = torch.stack([p['alpha'] for p in expert_params], dim=1)
        beta_stack = torch.stack([p['beta'] for p in expert_params], dim=1)

        G_expanded = G.unsqueeze(-1)
        w_weighted = torch.sum(G_expanded * w_stack, dim=1)
        params_w = F.softmax(w_weighted, dim=-1)
        alpha_weighted = torch.sum(G_expanded * alpha_stack, dim=1)
        beta_weighted = torch.sum(G_expanded * beta_stack, dim=1)

        if self.training:
            noise_scale = 0.02
            alpha_weighted = alpha_weighted + torch.randn_like(alpha_weighted) * noise_scale
            beta_weighted = beta_weighted + torch.randn_like(beta_weighted) * (noise_scale * 5.0)

        params = {
            'w': params_w,
            'alpha': F.softplus(alpha_weighted).clamp(0.1, 10.0),
            'beta': F.softplus(beta_weighted).clamp(1.0, 200.0)
        }

        L_ent = -(G * (G + 1e-8).log()).sum(-1).mean()
        extra_losses = {'L_ent': L_ent}  # raw, unscaled — NLLELGE applies the weight, not here

        # Load-balancing: penalizes the gate for being confident in whichever expert
        # currently dominates (Switch-Transformer style), computed per-step so it works
        # even at batch_size=1 — usage balances out across the dataset over many steps,
        # not within a single batch.
        with torch.no_grad():
            hard_choice = F.one_hot(G.argmax(dim=-1), num_classes=self.E).float()
        extra_losses['L_balance'] = self.E * (hard_choice * G).sum(dim=-1).mean()  # also raw

        if return_interpret:
            # Everything a separate plotting script needs, per forward call — no
            # saving to disk here, that's left to that script.
            extra_losses['expert_atts'] = expert_atts        # list of E tensors, each B x L
            extra_losses['patch_order'] = patch_order        # B x L (or None), maps position -> original patch index
            extra_losses['gate'] = G                         # B x E

        self._debug_count += 1
        if self._debug_count % 50 == 0:
            print(f"DEBUG Forward #{self._debug_count} - alpha std {params['alpha'].std():.4f}, "
                  f"beta std {params['beta'].std():.4f}, gate std {G.std():.4f}")
        return params, subloss, extra_losses

    def train_step(self, x_dict, t, c, constant_dict):
        params, subloss, extras = self(**x_dict)
        if torch.any(torch.isnan(params['alpha'])) or torch.any(torch.isnan(params['beta'])) or torch.any(torch.isnan(params['w'])):
            params['alpha'] = torch.ones_like(params['alpha']) * 2.0
            params['beta'] = torch.ones_like(params['beta']) * 10.0
            params['w'] = torch.ones_like(params['w']) / self.K

        surv = 1.0 - self.cdf(params, t)
        pdf_log = self.log_prob(params, t)
        pdf = torch.exp(pdf_log).clamp(min=1e-8, max=1e6)
        surv = surv.clamp(min=1e-8, max=1.0)
        return {
            't': t,
            'c': c,
            'pdf': pdf,
            'survival_func': surv,
            'subloss': subloss,
            'L_ent': extras['L_ent'],
            'L_balance': extras['L_balance'],
        }

    def eval_step(self, x_dict, t, c, constant_dict):
        with torch.no_grad():
            params, _, extras = self(**x_dict)
            B = t.size(0)
            device = t.device
            outputs = {"t": t, "c": c, "L_ent": extras['L_ent']}

            raw_eval_t = constant_dict["eval_t"].to(device)
            if raw_eval_t.dim() == 2 and raw_eval_t.size(0) == B:
                eval_t = raw_eval_t
            else:
                t_vec = raw_eval_t.reshape(-1)
                eval_t = t_vec.unsqueeze(0).expand(B, t_vec.numel()).contiguous()

            eval_t = eval_t.clamp(min=1e-6)

            median_t = torch.median(eval_t, dim=-1, keepdim=True)[0]
            cdf_median = self.cdf(params, median_t)
            surv_median = 1.0 - cdf_median
            risk_scores = -torch.log(surv_median.clamp(min=1e-12))

            cdf = self.cdf(params, eval_t)
            surv = 1.0 - cdf
            haz = -torch.log(surv.clamp(min=1e-12))

            haz_fixed = haz.clone()
            haz_fixed[:, 0] = risk_scores.squeeze(-1)
            outputs["cum_hazard_seqs"] = haz_fixed.transpose(0, 1).contiguous()

            t_min = constant_dict["t_min"].to(device)
            t_max = constant_dict["t_max"].to(device)
            num_int_steps = int(constant_dict["NUM_INT_STEPS"])
            grid = torch.linspace(float(t_min), float(t_max), num_int_steps, device=device)
            surv_grid = 1.0 - self.cdf(params, grid.unsqueeze(0).repeat(B, 1))
            outputs["survival_seqs"] = surv_grid.transpose(0, 1).contiguous()

            for eps in (0.1, 0.2, 0.3, 0.4, 0.5):
                key = f"t_max_{eps}"
                if key in constant_dict:
                    t_max_eps = constant_dict[key].to(device)
                    grid_eps = torch.linspace(float(t_min), float(t_max_eps), num_int_steps, device=device)
                    surv_eps = 1.0 - self.cdf(params, grid_eps.unsqueeze(0).repeat(B, 1))
                    outputs[f"survival_seqs_{eps}"] = surv_eps.transpose(0, 1).contiguous()

            outputs["eval_t"] = eval_t[0].detach().cpu()

            if not self._debug_printed:
                self._debug_printed = True
                print(f"DEBUG eval_step: median_t={median_t[0,0]:.4f}, risk min {risk_scores.min():.4f}, max {risk_scores.max():.4f}")

            return outputs

    def predict_step(self, x_dict):
        device = x_dict['x'].device
        with torch.no_grad():
            params, _, _ = self(**x_dict)
            t = torch.arange(0.1, 220.1, 0.1, device=device).unsqueeze(0)
            surv = 1.0 - self.cdf(params, t)
        return {'t': t.squeeze(0), 'p_survival': surv.squeeze(0)}
