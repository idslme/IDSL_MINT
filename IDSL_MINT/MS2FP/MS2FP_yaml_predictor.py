import yaml
import torch
import csv
import os
from multiprocessing import Pool
from joblib import Parallel, delayed
from tqdm import tqdm
from datetime import datetime

from IDSL_MINT.utils.MINT_logRecorder import MINT_logRecorder
from IDSL_MINT.utils.msp_file_utils import msp_deconvoluter, beam_search_tensor_generator
from IDSL_MINT.MS2FP.MS2FP_Model import MS2FP_Model
from IDSL_MINT.utils.msp_file_utils import MINT_address_check


def MS2FP_yaml_predictor(yaml_file):
    
    string_dict = {'none': None, 'true': True, 'false': False}

    with open(yaml_file, 'r') as yfle:
        yaml_data = yaml.safe_load(yfle)
        yfle.close()

    yaml_data = yaml_data['MINT_MS2FP_predictor']

    yaml_MSP_pickling = yaml_data['MSP']

    
    msp_file_directory = yaml_MSP_pickling['Directory to MSP files']
    msp_file_directory = MINT_address_check(msp_file_directory, address_check = True)

    msp_file_name = yaml_MSP_pickling['MSP files']
    MIN_MZ = float(yaml_MSP_pickling['Minimum m/z'])
    MAX_MZ = float(yaml_MSP_pickling['Maximum m/z'])
    INTERVAL_MZ = float(yaml_MSP_pickling['Interval m/z'])
    MIN_NUM_PEAKS = int(yaml_MSP_pickling['Minimum number of peaks'])
    MAX_NUM_PEAKS = int(yaml_MSP_pickling['Maximum number of peaks'])
    NOISE_REMOVAL_THRESHOLD = float(yaml_MSP_pickling['Noise removal threshold'])
    
    allowedSpectralEntropy = yaml_MSP_pickling['Allowed spectral entropy']
    if isinstance(allowedSpectralEntropy, str):
        allowedSpectralEntropy = string_dict.get(allowedSpectralEntropy.lower())
    
    number_processing_threads = int(yaml_MSP_pickling['Number processing threads'])
    
    Prediction_Parameters = yaml_data['Prediction Parameters']
    output_directory = Prediction_Parameters['Directory to store predictions']
    output_directory = MINT_address_check(output_directory, address_check = False)
    os.makedirs(output_directory, exist_ok = True)

    device = Prediction_Parameters['Device']
    if isinstance(device, str):
        device = device.lower()
        if device.__eq__("none"):
            device = None
        elif not (device.__eq__("cpu") or device.__eq__("cuda")):
            raise RuntimeError("Incorrect device type! Select None to automatically detect the processing device!")
    
    # To develop a device agnostic training loop
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    

    beam_size = int(Prediction_Parameters['Beam size'])

    logMINT = f"{output_directory}/logMINT_MS2FP_prediction.txt"
    MINT_logRecorder("=".join(["" for _ in range(100)]), logMINT = logMINT, allowedPrinting = True)
    MINT_logRecorder("Initiated processing the msp file(s)!", logMINT = logMINT, allowedPrinting = True)
    MINT_logRecorder(f"Initiation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n", logMINT = logMINT, allowedPrinting = True)

    export_parameters = (beam_size, device)

    deconvoluted_msp = msp_deconvoluter(msp_file_directory,
                                        msp_file_name,
                                        MIN_MZ,
                                        MAX_MZ,
                                        INTERVAL_MZ,
                                        MIN_NUM_PEAKS,
                                        MAX_NUM_PEAKS,
                                        NOISE_REMOVAL_THRESHOLD,
                                        allowedSpectralEntropy,
                                        export_parameters,
                                        number_processing_threads,
                                        logMINT,
                                        allowedPrinting = True)
    
    
    
    if len(deconvoluted_msp) > 0:
        
        MINT_logRecorder("Initiated creating tensors!", logMINT = logMINT, allowedPrinting = True)
    
        with Pool(processes = number_processing_threads) as P:
            beam_search_tuple = []
            with tqdm(total = len(deconvoluted_msp)) as Pbar:
                for results in P.imap(beam_search_tensor_generator, deconvoluted_msp):
                    if results:
                        beam_search_tuple.append(results)
                    Pbar.update()
                Pbar.close()
            P.close()
        
        del deconvoluted_msp
        
            
        MINT_logRecorder(f"\nDevice: {device}\n", logMINT = logMINT, allowedPrinting = True)
        MINT_logRecorder("\nLoading model weights!\n", logMINT = logMINT, allowedPrinting = True)
        Model_Parameters = yaml_data['Model Parameters']

        NUM_MZ_VOCABS = int(Model_Parameters['Number of m/z tokens'])
        D_MODEL = int(Model_Parameters['Dimension of model'])
        MZ_EMBED_MAX_NORM = float(Model_Parameters['Embedding norm of m/z tokens'])
        MZ_DROPOUT = float(Model_Parameters['Dropout probability of embedded m/z'])
        MAX_NUMBER_BITS = int(Model_Parameters['Maximum number of available fingerprint bits'])
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


        model = MS2FP_Model(NUM_MZ_VOCABS,
                            D_MODEL,
                            MZ_EMBED_MAX_NORM,
                            MZ_DROPOUT,
                            MAX_NUMBER_BITS,
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
        
        MINT_logRecorder("Initiated model prediction!", logMINT = logMINT, allowedPrinting = True)

        number_processing_threads = int(Prediction_Parameters['Number processing threads'])

        model.eval()
        if number_processing_threads == 1:
            name_msp_FP = []
            for x in tqdm(beam_search_tuple):
                with torch.inference_mode():
                    infer_x = model.beam_search_inference(x)
                
                name_msp_FP.append(infer_x)
        
        else:
            with torch.inference_mode():
                name_msp_FP = Parallel(n_jobs = number_processing_threads, timeout = 99999)(delayed(model.beam_search_inference)(x) for x in tqdm(beam_search_tuple))
        
        
        name_msp_FP.insert(0, ["MSP block name", "Finger Print"])
    
        name_csv = f"{output_directory}/IDSL_MINT_MS2FP_prediction.csv"

        MINT_logRecorder(f"\nPrediction results are stored at `{name_csv}`", logMINT = logMINT, allowedPrinting = True)

        with open(name_csv, "w", newline = '') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(name_msp_FP)
        
        MINT_logRecorder("=".join(["" for _ in range(100)]), logMINT = logMINT, allowedPrinting = True)
    
    else:
        MINT_logRecorder("No MSP block passed the criteria for model predcition!", logMINT = logMINT, allowedPrinting = True, warning_message = True)
    
    return None