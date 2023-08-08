import yaml
import torch
import os
import re
import pickle
import numpy as np
import pandas as pd
from multiprocessing import Pool
from joblib import Parallel, delayed
from tqdm import tqdm
from datetime import datetime

from IDSL_MINT.utils.MINT_logRecorder import MINT_logRecorder
from IDSL_MINT.FP2MS.FP2MS_Model import FP2MS_Model
from IDSL_MINT.utils.MINT_fingerprint_descriptor import MINT_fingerprint_descriptor
from IDSL_MINT.utils.msp_file_utils import MINT_address_check



def fingerPrintSreializer(list_smiles):
    
    nBitsToken = MINT_fingerprint_descriptor(list_smiles[0], list_smiles[1])
    
    return (nBitsToken, list_smiles[2], list_smiles[3])



def FP2MS_yaml_predictor(yaml_file):
    
    string_dict = {'none': None, 'true': True, 'false': False}

    with open(yaml_file, 'r') as yfle:
        yaml_data = yaml.safe_load(yfle)
        yfle.close()

    yaml_data = yaml_data['MINT_FP2MS_predictor']

    xlsx_address = yaml_data['Address to the reference XLXS file']

    MIN_MZ = float(yaml_data['Minimum m/z'])
    MAX_MZ = float(yaml_data['Maximum m/z'])
    INTERVAL_MZ = float(yaml_data['Interval m/z'])

    Fingerprints_parameters = yaml_data['Fingerprints']
    existing_fingerprint = Fingerprints_parameters['Use fingerprint row entries in the MSP blocks']
    MACCSKeys = Fingerprints_parameters['MACCS Keys']
    MorganFingerprint = Fingerprints_parameters['MorganFingerprint']

    count_fp = 0
    
    if existing_fingerprint:
        fingerprint_parameters = ('existing_fingerprint', None)
        count_fp += 1

    if MACCSKeys:
        fingerprint_parameters = ('MACCSKeys', None)
        count_fp += 1
    
    if MorganFingerprint:
        MorganFingerprint_parameters = Fingerprints_parameters['MorganFingerprint parameters']
        radius = MorganFingerprint_parameters['radius']
        nBits = MorganFingerprint_parameters['nBits']
        useChirality = MorganFingerprint_parameters['useChirality']
        if isinstance(useChirality, str):
            useChirality = string_dict.get(useChirality.lower())

        fingerprint_parameters = ('MorganFingerprint', None, radius, nBits, useChirality)
        count_fp += 1

    if count_fp != 1:
        raise RuntimeError("Please select `True` only for one Fingerprints method!")


    Prediction_Parameters = yaml_data['Prediction Parameters']

    output_directory = Prediction_Parameters['Directory to store predictions']
    output_directory = MINT_address_check(output_directory, address_check = False)
    os.makedirs(output_directory, exist_ok = True)

    device = Prediction_Parameters['Device']
    if isinstance(device, str):
        device = device.lower()
        if device.__eq__("none"):
            device = string_dict.get(device)
        elif not (device.__eq__("cpu") or device.__eq__("cuda")):
            raise RuntimeError("Incorrect device type! Select None to automatically detect the processing device!")
    
    if device is None:
        device = 'cpu'

    beam_size = int(Prediction_Parameters['Beam size'])

    number_processing_threads = int(Prediction_Parameters['Number processing threads'])

    logMINT = f"{output_directory}/logMINT_FP2MS_prediction.txt"
    MINT_logRecorder("=".join(["" for _ in range(100)]), logMINT = logMINT, allowedPrinting = True)
    MINT_logRecorder("Initiated processing the excel file!", logMINT = logMINT, allowedPrinting = True)
    MINT_logRecorder(f"Initiation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n", logMINT = logMINT, allowedPrinting = True)
    
    
    reference_xlsx = pd.read_excel(xlsx_address)

    headers = reference_xlsx.columns.tolist()

    Name_pattern = re.compile("^Name", re.IGNORECASE)
    SMILES_pattern = re.compile("^SMILES", re.IGNORECASE)
    InChI_string_pattern = re.compile("^INCHI", re.IGNORECASE)
    FingerPrint_string_pattern = re.compile("^FINGERPRINT", re.IGNORECASE)
    
    x_name = [i for i, string_head in enumerate(headers) if Name_pattern.search(string_head)]
    if len(x_name) != 1:
        print(f"\033[0;{'91'}m{f'WARNING!!! the `Name` coulmn header was not found in the Excel file!'}\033[0m")
    
    headers = [x for x in headers if not Name_pattern.match(x)]
    headers.insert(0, "Name")

    x_smiles = [i for i, string_head in enumerate(headers) if SMILES_pattern.search(string_head)]
    if len(x_smiles) > 1:
        raise RuntimeError("WARNING!!! redundant `SMILES` coulmn header!")
    
    x_inchi = [i for i, string_head in enumerate(headers) if InChI_string_pattern.search(string_head)]
    if len(x_inchi) > 1:
        raise RuntimeError("WARNING!!! redundant `INCHI` coulmn header!")
    
    x_fingerprint = [i for i, string_head in enumerate(headers) if FingerPrint_string_pattern.search(string_head)]
    if len(x_fingerprint) > 1:
        raise RuntimeError("WARNING!!! redundant `FINGERPRINT` coulmn header!")

    n_compounds = reference_xlsx.shape[0]

    MINT_logRecorder("Initiated calculating the fingerprints!", logMINT = logMINT, allowedPrinting = True)

    
    list_chemical_identifier_fingerprint = []
    
    for i in range(n_compounds):

        if 'SMILES' in headers:
            SMILES = reference_xlsx['SMILES'].iloc[i]
        else:
            SMILES = None
        
        if 'INCHI' in headers:
            InChI = reference_xlsx['INCHI'].iloc[i]
        else:
            InChI = None

        if 'FINGERPRINT' in headers:
            FINGERPRINT = reference_xlsx['FINGERPRINT'].iloc[i]
        else:
            FINGERPRINT = None
        
        chemical_identifiers = (SMILES, InChI, FINGERPRINT)
    
        list_chemical_identifier_fingerprint.append([chemical_identifiers, fingerprint_parameters, beam_size, device])

    Model_Parameters = yaml_data['Model Parameters']
    
    pkl_deconvoluted_msp_directory = Model_Parameters['Directory to load deconvoluted PKL file']
    pkl_deconvoluted_msp_directory = MINT_address_check(pkl_deconvoluted_msp_directory, address_check = True)
    
    mz_dictionary_name = f"{pkl_deconvoluted_msp_directory}/mz_dictionary.pkl"
    with open(mz_dictionary_name, "rb") as pkl:
        mz_dictionary = pickle.load(pkl)
    
    NUM_MZ_VOCABS = len(mz_dictionary)


    with Pool(processes = number_processing_threads) as P:
        beam_search_tuple = []
        for search_tuple in P.imap(fingerPrintSreializer, list_chemical_identifier_fingerprint):
            beam_search_tuple.append(search_tuple)
        P.close()
    
    MINT_logRecorder("\nLoading model weights!", logMINT = logMINT, allowedPrinting = True)

    D_MODEL = int(Model_Parameters['Dimension of model'])
    MZ_EMBED_MAX_NORM = float(Model_Parameters['Embedding norm of m/z tokens'])
    NUM_FINGERPRINT_VOCABS = int(Model_Parameters['Number of total fingerprint bits'])
    NUM_HEAD = int(Model_Parameters['Number of attention heads'])
    NUM_ENCODER_LAYER = int(Model_Parameters['Number of encoder layers'])
    NUM_DECODER_LAYER = int(Model_Parameters['Number of decoder layers'])
    TRANSFORMER_DROPOUT = float(Model_Parameters['Dropout probability of transformer'])
    TRANSFORMER_ACTIVATION = str(Model_Parameters['Activation function']).lower()

    if not TRANSFORMER_ACTIVATION in ('relu', 'gelu'):
        raise RuntimeError("Activation function must be 'relu' or 'gelu'!")
    
    if D_MODEL%NUM_HEAD != 0:
        raise RuntimeError("'Dimension of model' must be divisible to 'Number of attention heads'!")

    
    model = FP2MS_Model(NUM_MZ_VOCABS,
                        D_MODEL,
                        MZ_EMBED_MAX_NORM,
                        NUM_FINGERPRINT_VOCABS,
                        NUM_HEAD,
                        NUM_ENCODER_LAYER,
                        NUM_DECODER_LAYER,
                        TRANSFORMER_DROPOUT,
                        TRANSFORMER_ACTIVATION)

    model_address = Model_Parameters['Model address to load weights']
    model.load_state_dict(torch.load(f = model_address))
    
    if device.__eq__("cuda"):
        model.to(device)
    

    MINT_logRecorder("Initiated model prediction!\n\n", logMINT = logMINT, allowedPrinting = True)
    
    model.eval()
    if number_processing_threads == 1:
        msp_SMILES_job = []
        for x in tqdm(beam_search_tuple):
            with torch.inference_mode():
                infer_x = model.beam_search_inference(x)
            
            msp_SMILES_job.append(infer_x)
        
    else:
        with torch.inference_mode():
            msp_SMILES_job = Parallel(n_jobs = number_processing_threads, timeout = 99999)(delayed(model.beam_search_inference)(x) for x in tqdm(beam_search_tuple))

    mz_vocabs = np.arange(MIN_MZ, MAX_MZ, INTERVAL_MZ, dtype = float)

    msp_file = ""
    for i in range(n_compounds):

        spectra = msp_SMILES_job[i]

        if spectra is not None:
            for header in headers:
                msp_file = f"{msp_file}{header}: {reference_xlsx[header].iloc[i]}\n"

            num_peaks = len(spectra)
            msp_file = f"{msp_file}Num Peaks: {num_peaks}\n"

            for j in range(num_peaks):
                msp_file = f"{msp_file}{mz_vocabs[mz_dictionary[spectra[j]]]} 100\n"

            msp_file = f"{msp_file}\n\n"
        
        else:
            MINT_logRecorder(f"Invalid fingerprint value for entry of {reference_xlsx['Name'].iloc[i]}", logMINT = logMINT, allowedPrinting = True, warning_message = True)

    
    name_msp = f"{output_directory}/IDSL_MINT_FP2MS_prediction.msp"
    
    MINT_logRecorder(f"\nPrediction results are stored at `{name_msp}`", logMINT = logMINT, allowedPrinting = True)

    with open(name_msp, "w") as txtfile:
        txtfile.write(msp_file)
        txtfile.close()
    
    MINT_logRecorder("=".join(["" for _ in range(100)]), logMINT = logMINT, allowedPrinting = True)

    return None