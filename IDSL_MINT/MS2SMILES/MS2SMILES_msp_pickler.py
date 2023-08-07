import os, pickle
from typing import List, Union
from multiprocessing import Pool
from tqdm import tqdm

from IDSL_MINT.utils.MINT_logRecorder import MINT_logRecorder
from IDSL_MINT.utils.msp_file_utils import MINT_address_check, msp_deconvoluter
from IDSL_MINT.MS2SMILES.MS2SMILES_Assembler import getTrainingSMILESseqTokens


def MS2SMILES_msp_pickler(pkl_deconvoluted_msp_directory: str = "",
                          msp_file_directory: str = "",
                          msp_file_name: Union[str, List[str]] = "",
                          MIN_MZ: float = 50,
                          MAX_MZ: float = 1000,
                          INTERVAL_MZ: float = 0.01,
                          MIN_NUM_PEAKS: int = 4,
                          MAX_NUM_PEAKS: int = 512,
                          NOISE_REMOVAL_THRESHOLD: float = 0.01,
                          allowedSpectralEntropy: bool = False,
                          MAX_SMILES_sequence_length: int = 250,
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
    MINT_logRecorder(f"Spectral Entropy: {allowedSpectralEntropy}", logMINT = logMINT, allowedPrinting = False)
    MINT_logRecorder(f"MAX_SMILES_sequence_length: {MAX_SMILES_sequence_length}", logMINT = logMINT, allowedPrinting = False)
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
                                        MAX_SMILES_sequence_length,
                                        number_processing_threads,
                                        logMINT,
                                        allowedPrinting = True)
    
    ##
    ##########################################################################################################################
    ##

    MINT_logRecorder(f"\nSequencing SMILES and InChI chemical identifiers!\n", logMINT = logMINT, allowedPrinting = True)

    with Pool(processes = number_processing_threads) as P:
        mspTrainingSet = []
        with tqdm(total = len(deconvoluted_msp)) as Pbar:
            for results in P.imap(getTrainingSMILESseqTokens, deconvoluted_msp):
                mspTrainingSet.append(results)
                Pbar.update()
            Pbar.close()
        P.close()
    
    del deconvoluted_msp

    mspTrainingList_name = f"{pkl_deconvoluted_msp_directory}/MS2SMILES_TrainingSet.pkl"

    with open(mspTrainingList_name, "wb") as pkl:
        pickle.dump(mspTrainingSet, pkl)

    MINT_logRecorder(f"\nTraining data from .msp file(s) were stored at `{pkl_deconvoluted_msp_directory}`!", logMINT = logMINT, allowedPrinting = True)
    MINT_logRecorder("=".join(["" for _ in range(100)]), logMINT = logMINT, allowedPrinting = True)
    
    return None