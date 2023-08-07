import numpy as np
from IDSL_MINT.utils.SMILES_utils import getSMILESseqTokens
from IDSL_MINT.utils.MINT_logRecorder import MINT_logRecorder


def getTrainingSMILESseqTokens(deconvoluted_msp_block):

    SMILES_seq = None
    output = None
    
    i = deconvoluted_msp_block[0]
    nPeaks = deconvoluted_msp_block[2]
    
    if nPeaks > 0:
        
        trainingList = deconvoluted_msp_block[4]
        chemical_identifiers = trainingList[0]
        
        tokenized_PrecursorMZ = trainingList[1]
        if tokenized_PrecursorMZ:
            SMILES_seq, _ = getSMILESseqTokens(chemical_identifiers)


        logMINT = deconvoluted_msp_block[3]
        
        if SMILES_seq is None:
            
            msp_block_name = deconvoluted_msp_block[1]
            
            MINT_logRecorder(f"WARNING!!! Invalid chemical identifier for MSP block ID `{i}` related to `{msp_block_name}`!", logMINT = logMINT, allowedPrinting = False, warning_message = True)

        else:
            
            ## m/z-int spectra
            tokenized_PrecursorMZ = np.array([tokenized_PrecursorMZ])
            tokenized_MZ = np.array(trainingList[2])

            mz_tokens = np.concatenate((tokenized_PrecursorMZ, tokenized_MZ), axis = 0)
            int_tokens = np.concatenate((np.array([trainingList[3]]), np.array(trainingList[4])), axis = 0)

            ## SMILES
            max_sequence_length = trainingList[5]
            token_count = len(SMILES_seq)
            
            n_padded_tokens = max_sequence_length - token_count - 1
            if n_padded_tokens > 0:
                seqSMILESpadded = np.zeros(shape = (n_padded_tokens,)) # 0 is the token number for padding

            elif n_padded_tokens == 0:
                seqSMILESpadded = np.array([])
                
            else:
                MINT_logRecorder(f"The `Maximum length of SMILES characters` requires `{-n_padded_tokens}` more SMILES character tokens!", logMINT = logMINT, allowedPrinting = True, warning_message = True)


            seqSMILES = np.concatenate((np.array([1]), SMILES_seq, np.array([2]), seqSMILESpadded), axis = 0) # The EOS token should be ahead of padding tokens.

            seqSMILESpaddingMask = np.logical_not(np.isin(seqSMILES, np.array([0]))[:-1])

            trainingSet = (mz_tokens, int_tokens, seqSMILES, seqSMILESpaddingMask)

            output = (i, nPeaks, trainingSet)
    
    return output



def levenshtein_distance(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]



def normalized_levenshtein_distance(str1, str2):

    levenshtein_dist = levenshtein_distance(str1, str2)
    max_length = max(len(str1), len(str2))
    normalized_dist = 1 - levenshtein_dist / max_length
    return normalized_dist



import torch

from IDSL_MINT.utils.SMILES_utils import SMILER

def MS2SMILES_string_similarity(predicted_seqSMILES, gTruth_seqSMILES, n_msp_batch):

    predicted_seqSMILES = torch.softmax(predicted_seqSMILES, dim = 2)
    predicted_seqSMILES = predicted_seqSMILES.argmax(dim = 2)
    
    SumTanSim = 0
    for i in range(n_msp_batch):
        predicted_smiles = SMILER(predicted_seqSMILES[i, :].numpy())
        targeted_smiles = SMILER(gTruth_seqSMILES[i, :].numpy())

        SumTanSim += normalized_levenshtein_distance(predicted_smiles, targeted_smiles)
    

    return SumTanSim