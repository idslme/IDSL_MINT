import torch
from torch.optim import Adam
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torchinfo import summary

import os, gc
from timeit import default_timer as timer
from datetime import datetime

from IDSL_MINT.MS2SMILES.MS2SMILES_Dataset import MS2SMILES_Dataset
from IDSL_MINT.MS2SMILES.MS2SMILES_DataLoader import MS2SMILES_DataLoader
from IDSL_MINT.utils.MINT_logRecorder import MINT_logRecorder
from IDSL_MINT.utils.plot_loss_curves import plot_loss_curves
from IDSL_MINT.MS2SMILES.MS2SMILES_Assembler import MS2SMILES_string_similarity


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: str):

    model.train()
    
    train_loss  = 0
    train_acc = 0
    spectra_counter = 0
    
    for MZ_Tokens, INT, seqSMILES, seqSMILESpaddingMask in dataloader:
        
        gc.collect()
        
        # To squeeze the dimension added by DataLoader
        MZ_Tokens = MZ_Tokens.squeeze(dim = 0).to(device)
        INT = INT.squeeze(dim = 0).to(device)
        seqSMILES = seqSMILES.squeeze(dim = 0).to(device)
        seqSMILESpaddingMask = seqSMILESpaddingMask.squeeze(dim = 0).to(device)
        
        n_msp_batch = MZ_Tokens.shape[0]
        
        if n_msp_batch > 1: # To shuffle msp blocks in a batch of DataLoader
            permutation = torch.randperm(n = n_msp_batch).to(device)
            MZ_Tokens = MZ_Tokens[permutation]
            INT = INT[permutation]
            seqSMILES = seqSMILES[permutation]
            seqSMILESpaddingMask = seqSMILESpaddingMask[permutation]
        
        
        inputSeqSMILES = seqSMILES[:, :-1].to(device)
        targetSeqSMILES = seqSMILES[:, 1:].to(device)

        optimizer.zero_grad()
        
        predictedSeqSMILES = model(MZ_Tokens, INT, inputSeqSMILES, seqSMILESpaddingMask)
        
        pred_seq_sml = predictedSeqSMILES.reshape(-1, predictedSeqSMILES.size(-1))
        tgt_seq_sml = targetSeqSMILES.reshape(-1).type(torch.long)

        del MZ_Tokens, INT, inputSeqSMILES, seqSMILESpaddingMask
        
        loss = loss_fn(pred_seq_sml, tgt_seq_sml)
        train_loss += loss.item()
        
        loss.backward(retain_graph = False)
        del loss

        optimizer.step()
        
        train_acc += MS2SMILES_string_similarity(predictedSeqSMILES.to('cpu'), seqSMILES.to('cpu'), n_msp_batch)
        del predictedSeqSMILES, seqSMILES
        
        spectra_counter += n_msp_batch
        
    train_loss /= spectra_counter
    train_acc /= spectra_counter
    train_acc *= 100

    return train_loss, train_acc, spectra_counter
    

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: str):
              
    model.eval()

    test_loss = 0
    test_acc = 0
    spectra_counter = 0
    
    with torch.inference_mode():
        for MZ_Tokens, INT, seqSMILES, seqSMILESpaddingMask in dataloader:
            
            gc.collect()
        
            # To squeeze the dimension added by DataLoader
            MZ_Tokens = MZ_Tokens.squeeze(dim = 0).to(device)
            INT = INT.squeeze(dim = 0).to(device)
            seqSMILES = seqSMILES.squeeze(dim = 0).to(device)
            seqSMILESpaddingMask = seqSMILESpaddingMask.squeeze(dim = 0).to(device)
            
            n_msp_batch = MZ_Tokens.shape[0]

            inputSeqSMILES = seqSMILES[:, :-1].to(device)
            targetSeqSMILES = seqSMILES[:, 1:].to(device)
            
            predictedSeqSMILES = model(MZ_Tokens, INT, inputSeqSMILES, seqSMILESpaddingMask)

            pred_seq_sml = predictedSeqSMILES.reshape(-1, predictedSeqSMILES.size(-1))
            tgt_seq_sml = targetSeqSMILES.reshape(-1).type(torch.long)

            del MZ_Tokens, INT, inputSeqSMILES, seqSMILESpaddingMask
            
            loss = loss_fn(pred_seq_sml, tgt_seq_sml)
            test_loss += loss.item()
            del loss
            
            test_acc += MS2SMILES_string_similarity(predictedSeqSMILES.to('cpu'), seqSMILES.to('cpu'), n_msp_batch)
            del predictedSeqSMILES, seqSMILES
            
            spectra_counter += n_msp_batch
                
    test_loss /= spectra_counter
    test_acc /= spectra_counter
    test_acc *= 100

    return test_loss, test_acc, spectra_counter
    

def MS2SMILES_train(model: torch.nn.Module,
                    device: str = None,
                    reset_model_weights: bool = True,
                    label_smoothing: float = 0,
                    learning_rate: float = 0.00001,
                    betas: tuple[float, float] = (0.9, 0.98),
                    epsilon: float = 1e-09,
                    epochs: int = 5,
                    pkl_deconvoluted_msp_directory: str = "",
                    max_number_ions_per_batch: int = 2000,
                    split_ratio: list[int, int] = [0.80, 0.20],
                    random_state: int = 67,
                    output_directory: str = "",
                    number_processing_threads: int = 1) -> None:
                    

    if reset_model_weights:
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)


    # To develop a device agnostic training loop
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    

    if device.__eq__("cpu"):
        torch.set_num_threads(number_processing_threads)
    else:
        # To compile the model in torch versions >= 2.0.0 for faster training
        try:
            model.to(device)
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model)
            torch.autograd.set_detect_anomaly(True)
        except:
            pass
    
    ## LOSS function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index = 0, reduction = 'sum', label_smoothing = label_smoothing)

    ## logging the parameters
    os.makedirs(output_directory, exist_ok = True)
    logMINT = f"{output_directory}/logMINT_PyTorch_training_progress.txt"
    MINT_logRecorder("=".join(["" for _ in range(100)]), logMINT = logMINT, allowedPrinting = True)
    MINT_logRecorder(f"Initiation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", logMINT = logMINT, allowedPrinting = True)
    MINT_logRecorder(f"PyTorch version: {torch.__version__}\n", logMINT = logMINT, allowedPrinting = True)
    MINT_logRecorder(f"{summary(model)}", logMINT = logMINT, allowedPrinting = False)
    MINT_logRecorder(f"\nCreating batches for training and testing steps!", logMINT = logMINT, allowedPrinting = True)

    MS2SMILES_training = MS2SMILES_DataLoader(pkl_deconvoluted_msp_directory = pkl_deconvoluted_msp_directory, max_number_ions_per_batch = max_number_ions_per_batch)

    custom_dataset = MS2SMILES_Dataset(MS2SMILES_training = MS2SMILES_training)
    # Set random seeds
    torch.manual_seed(random_state)
    train_custom_dataset, test_custom_dataset = random_split(custom_dataset, split_ratio)
    train_dataloader_custom = DataLoader(train_custom_dataset, batch_size = 1, shuffle = True)
    test_dataloader_custom = DataLoader(test_custom_dataset, batch_size = 1, shuffle = False)

    ## logging the parameters
    MINT_logRecorder(f"\nDirectory to load deconvoluted PKL file: {pkl_deconvoluted_msp_directory}", logMINT = logMINT, allowedPrinting = True)
    MINT_logRecorder(f"Directory to store the trained model: {output_directory}\n", logMINT = logMINT, allowedPrinting = True)
    MINT_logRecorder(f"Number of training batches: {len(train_dataloader_custom)}", logMINT = logMINT, allowedPrinting = True)
    MINT_logRecorder(f"Number of testing batches: {len(test_dataloader_custom)}\n", logMINT = logMINT, allowedPrinting = True)
    MINT_logRecorder(f"Device: {device}", logMINT = logMINT, allowedPrinting = True)
    MINT_logRecorder(f"Label smoothing: {label_smoothing}", logMINT = logMINT, allowedPrinting = True)
    MINT_logRecorder(f"Learning rate: {learning_rate}", logMINT = logMINT, allowedPrinting = True)
    MINT_logRecorder(f"Betas: {betas}", logMINT = logMINT, allowedPrinting = True)
    MINT_logRecorder(f"Epsilon: {epsilon}", logMINT = logMINT, allowedPrinting = True)
    MINT_logRecorder(f"Designated number of loops: {epochs}\n", logMINT = logMINT, allowedPrinting = True)
    MINT_logRecorder(f"Maximum number of ions per batch: {max_number_ions_per_batch}", logMINT = logMINT, allowedPrinting = True)
    MINT_logRecorder(f"Random state: {random_state}", logMINT = logMINT, allowedPrinting = True)
    if device.__eq__("cpu"):
        MINT_logRecorder(f"Number of CPU processing threads: {number_processing_threads}", logMINT = logMINT, allowedPrinting = True)

    MINT_logRecorder(f"\nInitiated training!", logMINT = logMINT, allowedPrinting = True)

    start_time = timer()
    
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    epoch_time0 = timer()
    minimum_train_loss = float('inf')
    
    optimizer = Adam(model.parameters(), lr = learning_rate, betas = betas, eps = epsilon)
    
    accuracy_title = "Normalized Levenshtein Similarity for Predicted SMILES [0-100]"
    
    for epoch in range(epochs):

        train_loss, train_acc, train_counter = train_step(model = model,
                                                          dataloader = train_dataloader_custom,
                                                          loss_fn = loss_fn,
                                                          optimizer = optimizer,
                                                          device = device)
        
        
        test_loss, test_acc, test_counter = test_step(model = model,
                                                      dataloader = test_dataloader_custom,
                                                      loss_fn = loss_fn,
                                                      device = device)
        

        epoch_time1 = timer()
        elapsed_time = (epoch_time1 - epoch_time0)/3600
        epoch_time0 = epoch_time1
        
        MINT_logRecorder(f"Epoch: {epoch} | elapsed_time: {elapsed_time:.4f} hr | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f}% | train_spectra_count: {train_counter} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}% | test_spectra_count: {test_counter}", logMINT = logMINT, allowedPrinting = True)
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if train_loss < minimum_train_loss:
            minimum_train_loss = train_loss

            if epoch != 1:
                torch.save(obj = model.state_dict(), f = f"{output_directory}/MINT_MS2SMILES_model.pth") # only to save the state_dict() learned parameters

        plot_loss_curves(results, output_directory, accuracy_title)

    
    training_time = (timer() - start_time)/3600
    MINT_logRecorder(f"Total training time: {training_time:.2f} hr", logMINT = logMINT, allowedPrinting = True)
    MINT_logRecorder("=".join(["" for _ in range(100)]), logMINT = logMINT, allowedPrinting = True)

    return None