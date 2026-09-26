#from .TransMIL import TransMIL
#from .AMIL import AMIL
from .DSMIL import DSMILSurvival
#from .CLAM import CLAM_SB, CLAM_SB_Survival
from .SGCMLL import SGCMLL
from .EGMLL import EGMLL

backbone_cls = {
    #'AMIL':AMIL,
    'DSMIL':DSMILSurvival,
    #'TransMIL':TransMIL,
    #'CLAM-SB':CLAM_SB_Survival,
    "SGCMLL":SGCMLL
}

def create_WSI_model(cfg):
    if cfg.model.model_name == 'SurvivalEGMLL':
        cfg.model.as_backbone = True
        cfg.model.backbone = backbone_cls[cfg.model.backbone_name](**dict(cfg.model))
        model = EGMLL(**dict(cfg.model))
    else:
        raise NotImplementedError
    return model
