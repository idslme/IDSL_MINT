import os, shutil
from pathlib import Path
from tqdm import tqdm
import pickle
import torch
import numpy as np

from IDSL_MINT.utils.MINT_aggregate import MINT_peak_aggregate
from IDSL_MINT.utils.msp_file_utils import MINT_address_check

def MS2SMILES_DataLoader(pkl_deconvoluted_msp_directory, max_number_ions_per_batch):

    pkl_deconvoluted_msp_directory = MINT_address_check(pkl_deconvoluted_msp_directory, address_check = True)

    try:
        MS2SMILES_training = f"{pkl_deconvoluted_msp_directory}/MS2SMILES_training"
        if Path(MS2SMILES_training).is_dir():
            shutil.rmtree(MS2SMILES_training)
        
        os.makedirs(MS2SMILES_training, exist_ok = False)
    except:
        raise TypeError(f"Can't remove/create `{MS2SMILES_training}`!")


    mspTrainingSet_name = Path(f"{pkl_deconvoluted_msp_directory}/MS2SMILES_TrainingSet.pkl")
    if Path(mspTrainingSet_name).is_file():
            with open(mspTrainingSet_name, "rb") as pkl:
                mspTrainingSet = pickle.load(pkl)
        
    else:
        raise FileNotFoundError(f"Can't find `{mspTrainingSet_name}`!")

    
    msp_block_indices = MINT_peak_aggregate(mspTrainingSet, max_number_ions_per_batch)


    for i in tqdm(range(len(msp_block_indices))):
        indices = msp_block_indices[i]
        
        tokenized_mz, tokenized_int, seqSMILES, seqSMILESpaddingMask = [], [], [], []
        
        for j in indices:
            tokenized_mz1, tokenized_int1, seqSMILES1, seqSMILESpaddingMask1 = mspTrainingSet[j][2]

            tokenized_mz.append(tokenized_mz1)
            tokenized_int.append(tokenized_int1)
            seqSMILES.append(seqSMILES1)
            seqSMILESpaddingMask.append(seqSMILESpaddingMask1)
        
        tokenized_mz = np.stack(tokenized_mz)
        tokenized_int = np.stack(tokenized_int)
        seqSMILES = np.stack(seqSMILES)
        seqSMILESpaddingMask = np.stack(seqSMILESpaddingMask)

        tokenized_int = np.expand_dims(tokenized_int, axis = 2) # Add this dimension for the concatenation in the peak embedding step

        tokenized_mz = torch.tensor(tokenized_mz, dtype = torch.int)
        tokenized_int = torch.tensor(tokenized_int, dtype = torch.float32)
        seqSMILES = torch.tensor(seqSMILES, dtype = torch.long)
        seqSMILESpaddingMask = torch.tensor(seqSMILESpaddingMask, dtype = torch.float32)

        if seqSMILES.dim() == 1:
            seqSMILES = seqSMILES.unsqueeze(dim = 0)

        training_tensors_name = f"{MS2SMILES_training}/{indices[0]}_training_tensors.pth"
        torch.save({'tokenized_mz': tokenized_mz,
                    'tokenized_int': tokenized_int,
                    'seqSMILES': seqSMILES,
                    'seqSMILESpaddingMask': seqSMILESpaddingMask},
                    training_tensors_name)

    return MS2SMILES_training