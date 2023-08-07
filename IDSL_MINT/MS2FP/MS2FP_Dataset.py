from pathlib import Path
import torch
from torch.utils.data import Dataset
from typing import Tuple

class MS2FP_Dataset(Dataset):
    def __init__(self, MS2FP_training) -> None:

        self.paths = list(Path(MS2FP_training).glob("*_training_tensors.pth"))
    
    def load_pth(self, index: int):

        pathTorchTensors = self.paths[index]
        with open(pathTorchTensors, "rb") as pth:
            NLP_tensor = torch.load(pth)
            pth.close()
        
        return NLP_tensor['tokenized_mz'], NLP_tensor['tokenized_int'], NLP_tensor['nBitsTokens'], NLP_tensor['nBitsKeyPaddingMask']
        
    
    def __len__(self) -> int:
        return len(self.paths)

    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        tokenized_mz, tokenized_int, nBitsTokens, nBitsKeyPaddingMask = self.load_pth(index)
        
        return tokenized_mz, tokenized_int, nBitsTokens, nBitsKeyPaddingMask