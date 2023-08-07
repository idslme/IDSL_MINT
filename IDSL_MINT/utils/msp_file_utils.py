import re
import numpy as np
from pathlib import Path
from typing import List, Union, Tuple
from multiprocessing import Pool
from tqdm import tqdm


from IDSL_MINT.utils.spectral_entropy_utils import spectral_entropy_calculator
from IDSL_MINT.utils.MINT_logRecorder import MINT_logRecorder


Name_pattern = re.compile("^Name: ", re.IGNORECASE)
PrecursorMZ_pattern =  re.compile("^PrecursorMZ:", re.IGNORECASE)
InChI_pattern = re.compile("^InChI:", re.IGNORECASE)
SMILES_pattern = re.compile("^SMILES:", re.IGNORECASE)
num_peaks_pattern = re.compile("^Num Peaks:", re.IGNORECASE)
spaceSeperatingString_pattern = re.compile("\s")
InChI_string_pattern = re.compile("^InChI=", re.IGNORECASE)
Fingerprint_pattern = re.compile("^Fingerprint:", re.IGNORECASE)
invalidCharacters = re.compile('^"|"$| ')



def MINT_address_check(address: str = "",
                       address_check: bool = False) -> str:

    address = address.replace("\\", "/")

    while address[-1].__eq__("/"):
        address = address[:-1]

    if address_check:
        if not Path(address).is_dir():
            raise FileNotFoundError(f"`{address}` not found!")

    return address



def row_entry_selector(entry: List):
    
    if len(entry) > 1:
        l = [len(x) for x in entry]
        l = np.array(l)
        x_l = np.argmax(l)
        entry = entry[x_l]
    
    else:
        entry = entry[0]
    
    return entry



def msp_reader(msp_file_directory: str,
               msp_file_name: Union[str, List[str]] = "",
               logMINT: str = None,
               allowedPrinting: bool = True) -> List:
    
    msp_file_name = MINT_address_check(msp_file_name)
    
    if isinstance(msp_file_name, str):
        msp_file_name = [msp_file_name]

    msp = [""]
    for imsp in msp_file_name:
        
        msp_file_str = f"{msp_file_directory}/{imsp}"
        
        if Path(msp_file_str).is_file():
            MINT_logRecorder(f"Reading `{imsp}`!", logMINT = logMINT, allowedPrinting = allowedPrinting)
            with open(msp_file_str, "r") as file:
                msp.extend(file.readlines())
                file.close()

            msp.append("")
        
        else:
            raise FileNotFoundError(f"`{imsp}` is not available in the folder!")


    nRowsMSP = len(msp)
    ind = np.zeros(nRowsMSP, dtype = int)
    counter = 0
    for n in range(nRowsMSP):
        msp[n] = msp[n].replace("\n", "")
        if (msp[n] == "") or (msp[n] == " ") or (msp[n] == "  ") or (msp[n] == "\t"):
            ind[counter] = n
            counter += 1
    
    
    ind[counter] = nRowsMSP
    ind = ind[:(counter + 1)]
    index_diff = np.flatnonzero(np.diff(ind) > 1)
    x1_index = ind[index_diff] + 1
    x2_index = ind[index_diff + 1]

    return msp, x1_index, x2_index



def msp_block_deconvoluter(msp_block_object) -> Tuple:
    
    
    i = msp_block_object[0]
    msp_block = msp_block_object[1]
    MIN_NUM_PEAKS = msp_block_object[2]
    MAX_NUM_PEAKS = msp_block_object[3]
    NOISE_REMOVAL_THRESHOLD = msp_block_object[4]
    allowedSpectralEntropy = msp_block_object[5]
    mass_error = msp_block_object[6]
    mz_vocabs = msp_block_object[7]
    export_parameters = msp_block_object[8]
    logMINT = msp_block_object[9]


    msp_block_name = re.sub(Name_pattern, "", msp_block[0])
    
    trainingList = None

    ## Num Peaks
    num_peaks_row = [j for j, s in enumerate(msp_block) if num_peaks_pattern.search(s)]
    if num_peaks_row:
        
        nPeaks = len(msp_block) - num_peaks_row[0]
        if nPeaks >= MIN_NUM_PEAKS:

            msp_block = MoNA_comments_deconvolution(msp_block)
            msp_block = NIST_comment_deconvolution(msp_block)
            num_peaks_row = [j for j, s in enumerate(msp_block) if num_peaks_pattern.search(s)]

            ## PrecursorMZ
            PrecursorMZcheck = False
            PrecursorMZ = [s for s in msp_block if PrecursorMZ_pattern.search(s)]
            if PrecursorMZ:
                PrecursorMZ = row_entry_selector(PrecursorMZ)
                nCharPrecursorMZ = len(PrecursorMZ)
                if nCharPrecursorMZ > 13:
                    try:
                        PrecursorMZ = float(PrecursorMZ[12:nCharPrecursorMZ])
                        PrecursorMZcheck = PrecursorMZ >= mz_vocabs[0] and PrecursorMZ <= mz_vocabs[-1]
                    except:
                        PrecursorMZ = None
                
            
            if PrecursorMZcheck:
                
                ## SMILES
                SMILES = [s for s in msp_block if SMILES_pattern.search(s)]
                if SMILES:
                    SMILES = row_entry_selector(SMILES)
                    nCharSMILES = len(SMILES)
                    if nCharSMILES > 9:
                        SMILES = SMILES[8:]
                        SMILES = re.sub(invalidCharacters, "", SMILES)
                        
                    else:
                        SMILES = None
                else:
                    SMILES = None
                
                ## InChI
                InChI = [s for s in msp_block if InChI_pattern.search(s)]
                if InChI:
                    InChI = row_entry_selector(InChI)
                    nCharInChI = len(InChI)
                    if nCharInChI > 8:
                        InChI = InChI[7:]
                        InChI = re.sub(invalidCharacters, "", InChI)

                        if not InChI_string_pattern.search(InChI):
                            InChI = f"InChI={InChI}"
                    else:
                        InChI = None
                else:
                    InChI = None

                ## Finger print bits
                mol_finger_print = [s for s in msp_block if Fingerprint_pattern.search(s)]
                if mol_finger_print:
                    mol_finger_print = row_entry_selector(mol_finger_print)
                    nCharFP = len(mol_finger_print)
                    if nCharFP > 12:
                        mol_finger_print = mol_finger_print[13:]
                        mol_finger_print = re.sub(invalidCharacters, "", mol_finger_print)
                        
                    else:
                        mol_finger_print = None
                else:
                    mol_finger_print = None
                
                chemical_identifiers = (SMILES, InChI, mol_finger_print)

                num_peaks_row = num_peaks_row[0]
                nRowMSPblock = len(msp_block)

                if (nRowMSPblock - num_peaks_row) >= MIN_NUM_PEAKS: # To remove blocks with no significant peaks
                    
                    if spaceSeperatingString_pattern.search(msp_block[num_peaks_row + 1]):
                        
                        msp_block_mz = []
                        msp_block_int = []
                        for k in range((num_peaks_row + 1), nRowMSPblock):
                            nRowPeak = spaceSeperatingString_pattern.split(msp_block[k])

                            msp_block_mz.append(float(nRowPeak[0]))
                            msp_block_int.append(float(nRowPeak[1]))
                        

                        msp_block_mz = np.array(msp_block_mz, dtype = float)
                        msp_block_int = np.array(msp_block_int, dtype = float)
                        msp_block_int = msp_block_int/np.amax(msp_block_int)
                        
                        msp_block_mz, msp_block_int = spectra_integrator(msp_block_mz, msp_block_int, mass_error)
                        
                        ## To apply noise removal threshold
                        xNonNoise = np.flatnonzero(msp_block_int >= NOISE_REMOVAL_THRESHOLD)
                        nPeaks = len(xNonNoise)
                        if nPeaks >= MIN_NUM_PEAKS: # To remove blocks with no significant peaks
                            
                            msp_block_mz = msp_block_mz[xNonNoise]
                            msp_block_int = msp_block_int[xNonNoise]
                            
                            orderINT = np.argsort(msp_block_int, kind = 'mergesort')[::-1]

                            if nPeaks > MAX_NUM_PEAKS:
                                nPeaks = MAX_NUM_PEAKS
                                orderINT = orderINT[:MAX_NUM_PEAKS]
                            
                            msp_block_mz = msp_block_mz[orderINT]
                            msp_block_int = msp_block_int[orderINT]

                            x_MZ = np.flatnonzero((msp_block_mz >= mz_vocabs[0]) & (msp_block_mz <= mz_vocabs[-1]))
                            nPeaks2 = len(x_MZ)

                            if nPeaks2 >= nPeaks*0.90 and nPeaks2 >= MIN_NUM_PEAKS:

                                nPeaks = nPeaks2
                                msp_block_mz = msp_block_mz[x_MZ]
                                msp_block_int = msp_block_int[x_MZ]
                                
                                
                                tokenized_mz = []
                                for mz in msp_block_mz:
                                    token_id = np.argmin(abs(mz - mz_vocabs))
                                    tokenized_mz.append(token_id)

                                tokenized_precursormz = np.argmin(abs(PrecursorMZ - mz_vocabs))
                                
                                
                                if allowedSpectralEntropy:
                                    msp_spectra = np.column_stack((msp_block_mz, msp_block_int))
                                    spectral_entropy, msp_spectra = spectral_entropy_calculator(msp_spectra, allowedWeightedSpectralEntropy = True)
                                    unit_sum_int = msp_spectra[:, 1]
                                    trainingList = (chemical_identifiers, tokenized_precursormz, tokenized_mz, spectral_entropy, unit_sum_int, export_parameters)

                                else:
                                    trainingList = (chemical_identifiers, tokenized_precursormz, tokenized_mz, 2, msp_block_int, export_parameters)


    if trainingList is None:
        nPeaks = 0
        MINT_logRecorder(f"WARNING!!! Removed MSP block ID `{i}` related to `{msp_block_name}`!", logMINT = logMINT, allowedPrinting = False, warning_message = True)
        msp_block_name = None
        logMINT = None

    return i, msp_block_name, nPeaks, logMINT, trainingList



Comments_pattern = re.compile("^Comments: ", re.IGNORECASE)
Comments_Parent = re.compile("Parent=", re.IGNORECASE)

def MoNA_comments_deconvolution(mspBlock):

    Comments_Row, l = [], []
    for j, s in enumerate(mspBlock):
        if Comments_pattern.search(s):
            l.append(len(s))
            Comments_Row.append(j)

    if Comments_Row:
        if len(l) > 1:
            l = np.array(l)
            x_comments = np.argmax(l)
        else:
            x_comments = 0
        
        Comments_Row = Comments_Row[x_comments]
        Comments_Row = np.array(Comments_Row, dtype = int).item()
        Comments = mspBlock[Comments_Row]
        Comments = Comments[11:(len(Comments) - 1)] ## 11 is the length of `Comments: "`
        Comments = re.sub(Comments_Parent, "PrecursorMZ=", Comments)
        Comments = Comments.split('" "')

        mspBlock.remove(mspBlock[Comments_Row])
        Comments_Row -= 1

        for j in Comments:
            if "=" in j:
                first_index = j.index("=")
                j = f"{j[:first_index]}: {j[first_index + 1:]}"

                Comments_Row += 1
                mspBlock.insert(Comments_Row, j)
    
    return mspBlock



Comment_pattern = re.compile("^Comment: ", re.IGNORECASE)

def NIST_comment_deconvolution(mspBlock):

    Comment_Row, l = [], []
    for j, s in enumerate(mspBlock):
        if Comment_pattern.search(s):
            l.append(len(s))
            Comment_Row.append(j)

    if Comment_Row:
        if len(l) > 1:
            l = np.array(l)
            x_comment = np.argmax(l)
        else:
            x_comment = 0
        
        Comment_Row = Comment_Row[x_comment]
        Comment_Row = np.array(Comment_Row, dtype = int).item()
        Comment = mspBlock[Comment_Row]
        Comment = Comment[10:(len(Comment) - 1)] ## 10 is the length of `Comment: "`
        Comment = re.sub(Comments_Parent, "PrecursorMZ=", Comment)
        Comment = Comment.split('" "')

        mspBlock.remove(mspBlock[Comment_Row])
        Comment_Row -= 1

        for j in Comment:
            if "=" in j:
                first_index = j.index("=")
                j = f"{j[:first_index]}: {j[first_index + 1:]}"

                Comment_Row += 1
                mspBlock.insert(Comment_Row, j)
    
    return mspBlock



def spectra_integrator(raw_mz, raw_intensity, mass_error):

    orderIntensity = np.argsort(raw_intensity, kind = 'mergesort')[::-1]

    mz = np.zeros(raw_mz.shape[0])
    intensity = mz.copy()

    counter = 0

    for i in orderIntensity:

        if raw_mz[i] != 0:
            
            x_mz = np.where(abs(raw_mz - raw_mz[i]) <= mass_error)

            sum_intensity = np.sum(raw_intensity[x_mz])

            intensity[counter] = sum_intensity
            mz[counter] = np.sum(raw_mz[x_mz] * raw_intensity[x_mz]) / sum_intensity

            raw_intensity[x_mz] = 0
            raw_mz[x_mz] = 0

            counter += 1

    mz = mz[:counter]
    intensity = intensity[:counter]
    
    return mz, intensity



def msp_deconvoluter(msp_file_directory: str = "",
                     msp_file_name: Union[str, List[str]] = "",
                     MIN_MZ: float = 50,
                     MAX_MZ: float = 1000,
                     INTERVAL_MZ: float = 0.01,
                     MIN_NUM_PEAKS: int = 4,
                     MAX_NUM_PEAKS: int = 512,
                     NOISE_REMOVAL_THRESHOLD: float = 0.01,
                     allowedSpectralEntropy: bool = False,
                     export_parameters: Union[int, tuple] = None,
                     number_processing_threads: int = 1,
                     logMINT: str = None,
                     allowedPrinting: bool = False):
                     
    msp, x1_index, x2_index = msp_reader(msp_file_directory = msp_file_directory, msp_file_name = msp_file_name, logMINT = logMINT, allowedPrinting = allowedPrinting)

    mz_vocabs = np.arange(MIN_MZ, MAX_MZ, INTERVAL_MZ, dtype = float)
    
    MINT_logRecorder(f"\nInitiated serializing the msp blocks!\n", logMINT = logMINT, allowedPrinting = allowedPrinting)
    
    msp_block_objects = []
    for i in tqdm(range(len(x1_index))):
        msp_block = msp[x1_index[i]:x2_index[i]]
        msp_block_objects.append((i, msp_block, MIN_NUM_PEAKS, MAX_NUM_PEAKS, NOISE_REMOVAL_THRESHOLD, allowedSpectralEntropy, INTERVAL_MZ, mz_vocabs, export_parameters, logMINT))
    
    del msp, x1_index, x2_index, MIN_MZ, MAX_MZ, INTERVAL_MZ, MIN_NUM_PEAKS, MAX_NUM_PEAKS, NOISE_REMOVAL_THRESHOLD, mz_vocabs, export_parameters

    
    MINT_logRecorder(f"\nInitiated processing the msp blocks!\n", logMINT = logMINT, allowedPrinting = allowedPrinting)

    with Pool(processes = number_processing_threads) as P:
        deconvoluted_msp = []
        with tqdm(total = len(msp_block_objects)) as Pbar:
            for results in P.imap(msp_block_deconvoluter, msp_block_objects):
                deconvoluted_msp.append(results)
                Pbar.update()
            Pbar.close()
        P.close()
    
    del msp_block_objects
    
    return deconvoluted_msp



def beam_search_tensor_generator(msp_block_object):
    
    beam_search_tuple = None
    trainingList = msp_block_object[4]

    if trainingList:

        tokenized_PrecursorMZ = trainingList[1]

        if tokenized_PrecursorMZ:
            
            beam_size, device = trainingList[5][0], trainingList[5][1]
            msp_block_name = msp_block_object[1]
            
            tokenized_PrecursorMZ = np.array([tokenized_PrecursorMZ])
            tokenized_MZ = np.array(trainingList[2])

            mz_tokens = np.concatenate((tokenized_PrecursorMZ, tokenized_MZ), axis = 0)
            int_tokens = np.concatenate((np.array([trainingList[3]]), np.array(trainingList[4])), axis = 0)
            
            
            beam_search_tuple = (mz_tokens, int_tokens, beam_size, device, msp_block_name)

    return beam_search_tuple