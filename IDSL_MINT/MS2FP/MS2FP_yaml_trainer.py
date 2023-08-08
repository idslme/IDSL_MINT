import yaml, torch

from IDSL_MINT.MS2FP.MS2FP_msp_pickler import MS2FP_msp_pickler
from IDSL_MINT.MS2FP.MS2FP_Model import MS2FP_Model
from IDSL_MINT.MS2FP.MS2FP_training_function import MS2FP_train
from IDSL_MINT.utils.msp_file_utils import MINT_address_check


def MS2FP_yaml_trainer(yaml_file):
    
    string_dict = {'none': None, 'true': True, 'false': False}

    with open(yaml_file, 'r') as yfle:
        yaml_data = yaml.safe_load(yfle)
        yfle.close()

    yaml_data = yaml_data['MINT_MS2FP_trainer']

    yaml_MSP_pickling = yaml_data['MSP Pickling']

    perform_msp_pickling = yaml_MSP_pickling['Perform MSP pickling']
    if isinstance(perform_msp_pickling, str):
        perform_msp_pickling = string_dict.get(perform_msp_pickling.lower())

    if perform_msp_pickling:

        pkl_deconvoluted_msp_directory = yaml_MSP_pickling['Directory to store deconvoluted PKL file']
        pkl_deconvoluted_msp_directory = MINT_address_check(pkl_deconvoluted_msp_directory, address_check = False)

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
        
        MAX_NUMBER_BITS = int(yaml_MSP_pickling['Maximum number of available fingerprint bits'])
        Fingerprints_parameters = yaml_MSP_pickling['Fingerprints']
        existing_fingerprint = Fingerprints_parameters['Use fingerprint row entries in the MSP blocks']
        MACCSKeys = Fingerprints_parameters['MACCS Keys']
        MorganFingerprint = Fingerprints_parameters['MorganFingerprint']

        count_fp = 0
        if existing_fingerprint:
            fingerprint_parameters = ('existing_fingerprint', MAX_NUMBER_BITS)
            count_fp += 1

        if MACCSKeys:
            fingerprint_parameters = ('MACCSKeys', MAX_NUMBER_BITS)
            count_fp += 1
        
        if MorganFingerprint:
            MorganFingerprint_parameters = Fingerprints_parameters['MorganFingerprint parameters']
            radius = MorganFingerprint_parameters['radius']
            nBits = MorganFingerprint_parameters['nBits']
            useChirality = MorganFingerprint_parameters['useChirality']
            if isinstance(useChirality, str):
                useChirality = string_dict.get(useChirality.lower())

            fingerprint_parameters = ('MorganFingerprint', MAX_NUMBER_BITS, radius, nBits, useChirality)
            count_fp += 1

        if count_fp != 1:
            raise RuntimeError("Please select `True` only for one Fingerprints method!")

        number_processing_threads = int(yaml_MSP_pickling['Number processing threads'])

        MS2FP_msp_pickler(pkl_deconvoluted_msp_directory,
                            msp_file_directory,
                            msp_file_name,
                            MIN_MZ,
                            MAX_MZ,
                            INTERVAL_MZ,
                            MIN_NUM_PEAKS,
                            MAX_NUM_PEAKS,
                            NOISE_REMOVAL_THRESHOLD,
                            allowedSpectralEntropy,
                            fingerprint_parameters,
                            number_processing_threads)

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


    Training_Parameters = yaml_data['Training Parameters']

    output_directory = Training_Parameters['Directory to store the trained model']
    output_directory = MINT_address_check(output_directory, address_check = False)

    pkl_deconvoluted_msp_directory = Training_Parameters['Directory to load deconvoluted PKL file']
    pkl_deconvoluted_msp_directory = MINT_address_check(pkl_deconvoluted_msp_directory, address_check = True)

    reset_model_weights = Training_Parameters['Reset model weights']
    if isinstance(reset_model_weights, str):
        reset_model_weights = string_dict.get(reset_model_weights.lower())
    
    if not reset_model_weights:
        model_address = Training_Parameters['Model address to train']
        model.load_state_dict(torch.load(f = model_address))

    device = Training_Parameters['Device']
    if isinstance(device, str):
        device = device.lower()
        if device.__eq__("none"):
            device = None
        elif not (device.__eq__("cpu") or device.__eq__("cuda")):
            raise RuntimeError("Incorrect device type! Select None to automatically detect the processing device!")


    CrossEntropyLOSSfunction = Training_Parameters['Cross entropy LOSS function']
    label_smoothing = float(CrossEntropyLOSSfunction['Label smoothing'])

    Adam_optimizer_parameters = Training_Parameters['Adam optimizer function']
    learning_rate = float(Adam_optimizer_parameters['Learning rate'])
    betas = (float(Adam_optimizer_parameters['Beta1']), float(Adam_optimizer_parameters['Beta2']))
    epsilon = float(Adam_optimizer_parameters['Epsilon'])

    epochs = int(Training_Parameters['Maximum number of epochs'])
    max_number_ions_per_batch = int(Training_Parameters['Maximum number of ions per training step'])
    split_ratio = Training_Parameters['Split ratio between training and validation sets']
    random_state = int(Training_Parameters['Random state'])
    number_processing_threads = int(Training_Parameters['Number processing threads'])

    MS2FP_train(model,
                device,
                reset_model_weights,
                label_smoothing,
                learning_rate,
                betas,
                epsilon,
                epochs,
                pkl_deconvoluted_msp_directory,
                max_number_ions_per_batch,
                split_ratio,
                random_state,
                output_directory,
                number_processing_threads)
    
    return None