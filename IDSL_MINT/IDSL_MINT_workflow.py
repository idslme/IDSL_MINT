import argparse
import yaml

from IDSL_MINT.utils.msp_file_utils import MINT_address_check

from IDSL_MINT.MS2FP.MS2FP_yaml_trainer import MS2FP_yaml_trainer
from IDSL_MINT.MS2FP.MS2FP_yaml_predictor import MS2FP_yaml_predictor

from IDSL_MINT.MS2SMILES.MS2SMILES_yaml_trainer import MS2SMILES_yaml_trainer
from IDSL_MINT.MS2SMILES.MS2SMILES_yaml_predictor import MS2SMILES_yaml_predictor

from IDSL_MINT.FP2MS.FP2MS_yaml_trainer import FP2MS_yaml_trainer
from IDSL_MINT.FP2MS.FP2MS_yaml_predictor import FP2MS_yaml_predictor


def IDSL_MINT_workflow(yaml_file):
    
    yaml_file = MINT_address_check(yaml_file, address_check = False)

    with open(yaml_file, 'r') as yfle:
        yaml_data = yaml.safe_load(yfle)
        yfle.close()

    yaml_type = list(yaml_data.keys())[0]

    if yaml_type.__eq__('MINT_MS2FP_trainer'):
        MS2FP_yaml_trainer(yaml_file)
    
    elif yaml_type.__eq__('MINT_MS2FP_predictor'):
        MS2FP_yaml_predictor(yaml_file)
    
    elif yaml_type.__eq__('MINT_MS2SMILES_trainer'):
        MS2SMILES_yaml_trainer(yaml_file)
    
    elif yaml_type.__eq__('MINT_MS2SMILES_predictor'):
        MS2SMILES_yaml_predictor(yaml_file)

    elif yaml_type.__eq__('MINT_FP2MS_trainer'):
        FP2MS_yaml_trainer(yaml_file)

    elif yaml_type.__eq__('MINT_FP2MS_predictor'):
        FP2MS_yaml_predictor(yaml_file)
    
    else:
        raise RuntimeError('Incorrect type of `yaml` file!')


def MINT_workflow():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type = str, help = ' --> Path to the YAML file for IDSL_MINT data processing')
    args = parser.parse_args()

    IDSL_MINT_workflow(args.yaml)

if __name__ == '__main__':
    MINT_workflow()