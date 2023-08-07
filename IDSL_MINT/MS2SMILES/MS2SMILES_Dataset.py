from pathlib import Path
import torch
from torch.utils.data import Dataset
from typing import Tuple


class MS2SMILES_Dataset(Dataset):
    def __init__(self, MS2SMILES_training) -> None:

        self.paths = list(Path(MS2SMILES_training).glob("*_training_tensors.pth"))
    
    def load_pth(self, index: int):

        pathTorchTensors = self.paths[index]
        with open(pathTorchTensors, "rb") as pth:
            SMILES_tensor = torch.load(pth)
            pth.close()
        
        return (SMILES_tensor['tokenized_mz'],
                SMILES_tensor['tokenized_int'],
                SMILES_tensor['seqSMILES'],
                SMILES_tensor['seqSMILESpaddingMask'])
        
    
    def __len__(self) -> int:
        return len(self.paths)

    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.load_pth(index)