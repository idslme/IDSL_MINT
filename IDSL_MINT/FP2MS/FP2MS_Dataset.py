from pathlib import Path
import torch
from torch.utils.data import Dataset
from typing import Tuple


class FP2MS_Dataset(Dataset):
    def __init__(self, FP2MS_training) -> None:

        self.paths = list(Path(FP2MS_training).glob("*_training_tensors.pth"))
    
    def load_pth(self, index: int):

        pathTorchTensors = self.paths[index]
        with open(pathTorchTensors, "rb") as pth:
            FP_tensor = torch.load(pth)
            pth.close()
        
        return (FP_tensor['tokenized_MZ'], FP_tensor['FingerPrint'], FP_tensor['FingerPrintPaddingMask'])
        
    
    def __len__(self) -> int:
        return len(self.paths)

    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        return self.load_pth(index)