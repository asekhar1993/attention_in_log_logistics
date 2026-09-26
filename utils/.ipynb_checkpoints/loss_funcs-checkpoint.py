from typing import Any
import torch

def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, label, censorship, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, label, censorship, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, label, censorship, alpha=alpha)
        

#################################losss for log-logistic#######################
# class NLLELGE(object):
#     def __init__(self, lambda_ent=0.001):  # FIXED: Reduced from 0.01
#         self.lambda_ent = lambda_ent

#     def __call__(self, datas):
#         c = datas['c']
#         pdf = datas['pdf']
#         survival_func = datas['survival_func']

#         # Keep your original (correct) formulation
#         base_loss = -(1-c) * torch.log((pdf+1e-6)/(survival_func+1e-6)) - torch.log(survival_func + 1e-6)
        
#         reg_loss = -(self.lambda_ent * datas['L_ent']) if 'L_ent' in datas else 0

#         return base_loss.mean() + reg_loss

##################################loss for log -logistics amil in egmll 3##############################
class NLLELGE(object):
    def __init__(self, lambda_ent=0.01, lambda_balance=0.01):
        self.lambda_ent = lambda_ent
        self.lambda_balance = lambda_balance

    def __call__(self, datas):
        c = datas['c']
        pdf = datas['pdf']
        survival_func = datas['survival_func']

        base_loss = -(1-c) * torch.log((pdf+1e-6)/(survival_func+1e-6)) - torch.log(survival_func + 1e-6)

        reg_loss = 0
        if 'L_ent' in datas:
            # L_ent is entropy of the gate distribution -- HIGHER entropy = more spread
            # out, which is what we want, so we MAXIMIZE it (subtract it from the loss).
            reg_loss = reg_loss - self.lambda_ent * datas['L_ent']

        if 'L_balance' in datas:
            # L_balance is HIGH when the gate is confidently routing to one expert, so
            # we MINIMIZE it (add it to the loss, no negation).
            reg_loss = reg_loss + self.lambda_balance * datas['L_balance']

        return base_loss.mean() + reg_loss