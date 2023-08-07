import numpy as np

from IDSL_MINT.utils.MINT_fingerprint_descriptor import MINT_fingerprint_descriptor
from IDSL_MINT.utils.MINT_logRecorder import MINT_logRecorder

def getFingerPrintTokens(deconvoluted_msp_block):

    nBitsToken = None
    output = None

    i = deconvoluted_msp_block[0]
    nPeaks = deconvoluted_msp_block[2]

    if nPeaks > 0:
        
        trainingList = deconvoluted_msp_block[4]
        tokenized_PrecursorMZ = trainingList[1]

        if tokenized_PrecursorMZ:

            chemical_identifiers = trainingList[0]
            fingerprint_parameters = trainingList[5]

            nBitsToken = MINT_fingerprint_descriptor(chemical_identifiers, fingerprint_parameters)


        logMINT = deconvoluted_msp_block[3]
        
        if nBitsToken is None:
            
            msp_block_name = deconvoluted_msp_block[1]
            
            MINT_logRecorder(f"WARNING!!! Can't process molecular finger print tokens for MSP block ID `{i}` related to `{msp_block_name}`!", logMINT = logMINT, allowedPrinting = False, warning_message = True)

        else:
            
            ## m/z-int spectra
            tokenized_PrecursorMZ = np.array([tokenized_PrecursorMZ])
            tokenized_MZ = np.array(trainingList[2])

            mz_tokens = np.concatenate((tokenized_PrecursorMZ, tokenized_MZ), axis = 0)
            int_tokens = np.concatenate((np.array([trainingList[3]]), np.array(trainingList[4])), axis = 0)

            ## Finger print tokens
            max_sequence_length = fingerprint_parameters[1]
            nBitsToken += 3
            token_count = len(nBitsToken)
            
            n_padded_tokens = max_sequence_length - token_count - 1
            if n_padded_tokens > 0:
                nBitsTokenPadded = np.zeros(shape = (n_padded_tokens,)) # 0 is the token number for padding

            elif n_padded_tokens == 0:
                nBitsTokenPadded = np.array([])
                
            else:
                MINT_logRecorder(f"The `Maximum number of available fingerprint bits` requires `{-n_padded_tokens}` more fingerprint tokens!", logMINT = logMINT, allowedPrinting = True, warning_message = True)
            

            nBitsToken = np.concatenate((np.array([1]), nBitsToken, np.array([2]), nBitsTokenPadded), axis = 0) # The EOS token should be ahead of padding tokens.

            nBitsKeyPaddingMask = np.logical_not(np.isin(nBitsToken, np.array([0]))[:-1])

            trainingList = (mz_tokens, int_tokens, nBitsToken, nBitsKeyPaddingMask)
            output = (i, nPeaks, trainingList)
        
    
    return output



import torch

from IDSL_MINT.utils.MINT_tanimoto_coefficient import MINT_tanimoto_coefficient


def nBitsTokensTanimotoCoefficient(predict_finger_print, targets, n_msp_batch):

    predictions = torch.softmax(predict_finger_print, dim = 2)
    predictions = predictions.argmax(dim = 2)

    SumTanSim = 0
    for i in range(n_msp_batch):
        predicted_fingerprint = predictions[i, :].numpy()
        targeted_fingerprint = targets[i, :].numpy()

        SumTanSim += MINT_tanimoto_coefficient(predicted_fingerprint, targeted_fingerprint, remove_invalid_tokens = True)
    

    return SumTanSim