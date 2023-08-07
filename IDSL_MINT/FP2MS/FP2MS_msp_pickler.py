import pickle, os
import numpy as np
import pandas as pd

from typing import List, Union
from multiprocessing import Pool
from tqdm import tqdm

from IDSL_MINT.utils.MINT_logRecorder import MINT_logRecorder
from IDSL_MINT.utils.msp_file_utils import MINT_address_check, msp_deconvoluter
from IDSL_MINT.FP2MS.FP2MS_Assembler import getFingerPrintTokens2MS


def msp_token_update(deconvoluted_msp_block, sequential_tokens):

    if deconvoluted_msp_block is not None:

        n_peaks = deconvoluted_msp_block[1]
        peak_0, mz_0 = 0, []

        for j in range(n_peaks):
            
            if deconvoluted_msp_block[2][0][j] in sequential_tokens:
                
                peak_0 += 1
                mz_0.append(sequential_tokens[deconvoluted_msp_block[2][0][j]])
        
        if peak_0 > 0:
            deconvoluted_msp_block = (deconvoluted_msp_block[0],
                                      peak_0,
                                      (np.array(mz_0),
                                      deconvoluted_msp_block[2][1],
                                      deconvoluted_msp_block[2][2]))
        
        else:
            deconvoluted_msp_block = None
    
    return deconvoluted_msp_block


def parallel_msp_token_update(deconvoluted_msp_block_list):
    return msp_token_update(deconvoluted_msp_block_list[0], deconvoluted_msp_block_list[1])


def FP2MS_msp_pickler(pkl_deconvoluted_msp_directory: str = "",
                      msp_file_directory: str = "",
                      msp_file_name: Union[str, List[str]] = "",
                      MIN_MZ: float = 50,
                      MAX_MZ: float = 1000,
                      INTERVAL_MZ: float = 0.01,
                      MIN_NUM_PEAKS: int = 4,
                      MAX_NUM_PEAKS: int = 512,
                      NOISE_REMOVAL_THRESHOLD: float = 0.01,
                      allowedSpectralEntropy: bool = True,
                      fingerprint_parameters: Union[int, tuple] = 250,
                      number_top_mz: int = 10000,
                      number_processing_threads: int = 1) -> None:

    """
    Pickles MSP files using specified methods.

    Parameters:
        - pkl_deconvoluted_msp_directory: A string representing the directory where the pickled MSP files will be saved.
        - msp_file_directory: A string representing the directory where the MSP files are located.
        - msp_file_name: A string or list of strings representing the names of the MSP files to be pickled.
        - MIN_MZ: A float representing the minimum m/z value to include in the pickled data.
        - MAX_MZ: A float representing the maximum m/z value to include in the pickled data.
        - INTERVAL_MZ: A float representing the m/z interval to use when creating the pickled data.
        - MIN_NUM_PEAKS: An integer representing the minimum number of peaks required to include a spectrum in the pickled data.
        - MAX_NUM_PEAKS: An integer representing the embedding dimension to use when creating the pickled data.
        - NOISE_REMOVAL_THRESHOLD: A float representing the threshold value to use when removing noise from the data.
        - allowedSpectralEntropy: A boolean representing whether to use the allowed spectral entropy metric when creating the pickled data. Default is False.
        - fingerprint_parameters: fingerprint_parameters.
        - number_top_mz: Number top m/z
        - number_processing_threads: An integer representing the number of processing threads to use when creating the pickled data. Default is 1.

    Returns:

        This function does not return anything, but saves the pickled MSP files to disk.

    """
    
    msp_file_directory = MINT_address_check(address = msp_file_directory, address_check = True)

    pkl_deconvoluted_msp_directory = MINT_address_check(address = pkl_deconvoluted_msp_directory, address_check = False)
    os.makedirs(pkl_deconvoluted_msp_directory, exist_ok = True)


    ## logging the parameters
    logMINT = f"{pkl_deconvoluted_msp_directory}/logMINT_training_set_pkl.txt"
    MINT_logRecorder("=".join(["" for _ in range(100)]), logMINT = logMINT, allowedPrinting = True)
    MINT_logRecorder(f"\nInitiated separating mass spectra and sequencing chemical identifiers from the msp file(s)!", logMINT = logMINT, allowedPrinting = True)
    MINT_logRecorder(f"Variables used to generate training_set_pkl are: \n", logMINT = logMINT, allowedPrinting = False)
    MINT_logRecorder(f"msp_file_directory: {msp_file_directory}", logMINT = logMINT, allowedPrinting = False)
    MINT_logRecorder(f"pkl_deconvoluted_msp_directory: {pkl_deconvoluted_msp_directory}", logMINT = logMINT, allowedPrinting = False)
    MINT_logRecorder(f"msp_file_name: {msp_file_name}", logMINT = logMINT, allowedPrinting = False)
    MINT_logRecorder(f"MIN_MZ: {MIN_MZ}", logMINT = logMINT, allowedPrinting = False)
    MINT_logRecorder(f"MAX_MZ: {MAX_MZ}", logMINT = logMINT, allowedPrinting = False)
    MINT_logRecorder(f"INTERVAL_MZ: {INTERVAL_MZ}", logMINT = logMINT, allowedPrinting = False)
    MINT_logRecorder(f"MIN_NUM_PEAKS: {MIN_NUM_PEAKS}", logMINT = logMINT, allowedPrinting = False)
    MINT_logRecorder(f"MAX_NUM_PEAKS: {MAX_NUM_PEAKS}", logMINT = logMINT, allowedPrinting = False)
    MINT_logRecorder(f"NOISE_REMOVAL_THRESHOLD: {NOISE_REMOVAL_THRESHOLD}", logMINT = logMINT, allowedPrinting = False)
    MINT_logRecorder(f"Fingerprint parameters: {fingerprint_parameters}", logMINT = logMINT, allowedPrinting = False)
    MINT_logRecorder(f"number_processing_threads: {number_processing_threads}\n", logMINT = logMINT, allowedPrinting = False)
    
    ##
    ##########################################################################################################################
    ##

    deconvoluted_msp = msp_deconvoluter(msp_file_directory,
                                        msp_file_name,
                                        MIN_MZ,
                                        MAX_MZ,
                                        INTERVAL_MZ,
                                        MIN_NUM_PEAKS,
                                        MAX_NUM_PEAKS,
                                        NOISE_REMOVAL_THRESHOLD,
                                        allowedSpectralEntropy,
                                        fingerprint_parameters,
                                        number_processing_threads,
                                        logMINT,
                                        allowedPrinting = True)
    
    ##
    ##########################################################################################################################
    ##

    MINT_logRecorder(f"\nObtaining molecular fingerprints from MSP blocks!\n", logMINT = logMINT, allowedPrinting = True)

    with Pool(processes = number_processing_threads) as P:
        mspTrainingSet = []
        with tqdm(total = len(deconvoluted_msp)) as Pbar:
            for results in P.imap(getFingerPrintTokens2MS, deconvoluted_msp):
                mspTrainingSet.append(results)
                Pbar.update()
            Pbar.close()
        P.close()
    
    del deconvoluted_msp

    ##
    ##########################################################################################################################
    ##

    MINT_logRecorder(f"\nCreating dictionary of top frequent m/z!\n", logMINT = logMINT, allowedPrinting = True)

    all_tokens = []

    for i in range(len(mspTrainingSet)):
        if mspTrainingSet[i] is not None:
            all_tokens.extend(mspTrainingSet[i][2][0].tolist())

    series_tokens = pd.Series(all_tokens, name = 'Category')
    table_tokens = series_tokens.value_counts().reset_index()
    table_tokens.columns = ['Category', 'Frequency']

    if table_tokens.shape[0] < number_top_mz:
        number_top_mz = table_tokens.shape[0] - 1
    
    x5000 = np.flatnonzero(table_tokens['Frequency'] >= table_tokens['Frequency'].iloc[number_top_mz])
    mz_tokens = [table_tokens['Category'].iloc[i] for i in x5000]
    mz_tokens = sorted(mz_tokens)

    sequential_tokens = {token: i for i, token in enumerate(mz_tokens)}

    list_parallel_msp_token_update = []
    for msp_block_train in mspTrainingSet:
        list_parallel_msp_token_update.append([msp_block_train, sequential_tokens])


    with Pool(processes = number_processing_threads) as P:
        mspTrainingSet = []
        with tqdm(total = len(list_parallel_msp_token_update)) as Pbar:
            for msp_block in P.imap(parallel_msp_token_update, list_parallel_msp_token_update):
                if msp_block is not None:
                    mspTrainingSet.append(msp_block)
                Pbar.update()
            Pbar.close()
        P.close()
    

    ##
    ##########################################################################################################################
    ##
    
    mspTrainingList_name = f"{pkl_deconvoluted_msp_directory}/FP2MS_TrainingSet.pkl"

    with open(mspTrainingList_name, "wb") as pkl:
        pickle.dump(mspTrainingSet, pkl)
        
    mz_dictionary = {value: key for key, value in sequential_tokens.items()}
    
    mz_dictionary_name = f"{pkl_deconvoluted_msp_directory}/mz_dictionary.pkl"
    
    with open(mz_dictionary_name, "wb") as pkl:
        pickle.dump(mz_dictionary, pkl)
    
    MINT_logRecorder(f"\nm/z dictionary was stored at `{mz_dictionary_name}`!", logMINT = logMINT, allowedPrinting = True)
    MINT_logRecorder(f"\nTraining data from .msp file(s) were stored at `{pkl_deconvoluted_msp_directory}`!", logMINT = logMINT, allowedPrinting = True)
    MINT_logRecorder("=".join(["" for _ in range(100)]), logMINT = logMINT, allowedPrinting = True)
    
    return None