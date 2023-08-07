import os, shutil
from pathlib import Path
from tqdm import tqdm
import pickle
import torch
import numpy as np

from IDSL_MINT.utils.MINT_aggregate import MINT_peak_aggregate
from IDSL_MINT.utils.msp_file_utils import MINT_address_check

def FP2MS_DataLoader(pkl_deconvoluted_msp_directory, max_number_ions_per_batch):

    pkl_deconvoluted_msp_directory = MINT_address_check(pkl_deconvoluted_msp_directory, address_check = True)

    try:
        FP2MS_training = f"{pkl_deconvoluted_msp_directory}/FP2MS_training"
        if Path(FP2MS_training).is_dir():
            shutil.rmtree(FP2MS_training)
        
        os.makedirs(FP2MS_training, exist_ok = False)
    except:
        raise TypeError(f"Can't remove/create `{FP2MS_training}`!")


    mspTrainingSet_name = Path(f"{pkl_deconvoluted_msp_directory}/FP2MS_TrainingSet.pkl")
    if Path(mspTrainingSet_name).is_file():
        with open(mspTrainingSet_name, "rb") as pkl:
            mspTrainingSet = pickle.load(pkl)
    
    else:
        raise FileNotFoundError(f"Can't find `{mspTrainingSet_name}`!")

    
    msp_block_indices = MINT_peak_aggregate(mspTrainingSet, max_number_ions_per_batch)


    for i in tqdm(range(len(msp_block_indices))):
        indices = msp_block_indices[i]
        
        FingerPrint, tokenized_MZ, FingerPrintPaddingMask = [], [], []
        
        for j in indices:
            tokenized_MZ1, FingerPrint1, FingerPrintPaddingMask1 = mspTrainingSet[j][2]

            tokenized_MZ.append(tokenized_MZ1)
            FingerPrint.append(FingerPrint1)
            FingerPrintPaddingMask.append(FingerPrintPaddingMask1)
        
        
        tokenized_MZ = np.stack(tokenized_MZ)
        FingerPrint = np.stack(FingerPrint)
        FingerPrintPaddingMask = np.stack(FingerPrintPaddingMask)


        tokenized_MZ = torch.tensor(tokenized_MZ, dtype = torch.int)
        FingerPrint = torch.tensor(FingerPrint, dtype = torch.long)
        FingerPrintPaddingMask = torch.tensor(FingerPrintPaddingMask, dtype = torch.bool)

        if FingerPrint.dim() == 1:
            FingerPrint = FingerPrint.unsqueeze(dim = 0)

        training_tensors_name = f"{FP2MS_training}/{indices[0]}_training_tensors.pth"
        torch.save({'tokenized_MZ': tokenized_MZ,
                    'FingerPrint': FingerPrint,
                    'FingerPrintPaddingMask': FingerPrintPaddingMask},
                    training_tensors_name)

    return FP2MS_training