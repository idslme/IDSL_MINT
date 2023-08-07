import numpy as np

invalid_tokens = np.array([0, 1, 2])

def MINT_tanimoto_coefficient(Pred_FP, Target_FP, remove_invalid_tokens = True):
    
    Pred_FP = np.unique(Pred_FP)
    if remove_invalid_tokens:
        Pred_FP = Pred_FP[~np.isin(Pred_FP, invalid_tokens)]

    
    Target_FP = np.unique(Target_FP)
    if remove_invalid_tokens:
        Target_FP = Target_FP[~np.isin(Target_FP, invalid_tokens)]

    
    intersection = np.isin(Pred_FP, Target_FP).sum()

    union = Pred_FP.shape[0] + Target_FP.shape[0] - intersection

    if union == 0:
        tanimoto = 0
    else:
        tanimoto = intersection / union

    return tanimoto