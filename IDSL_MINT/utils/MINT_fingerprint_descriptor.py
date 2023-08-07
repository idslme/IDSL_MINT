import re
import numpy as np
from typing import Tuple

from rdkit.Chem import MolFromInchi, MolFromSmiles
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect, DataStructs


InChI_string_pattern = re.compile("^InChI", re.IGNORECASE)
invalidCharacters = re.compile('^"|"$| ')


# Disable RDKit warning messages
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def GetRDKitMorganFingerprintTokens(chemical_identifier: str,
                                    radius: int,
                                    nBits: int,
                                    useChirality: bool) -> np.ndarray:
    
    try:
        chemical_identifier = re.sub(invalidCharacters, "", chemical_identifier)
        
        if InChI_string_pattern.search(chemical_identifier):
            mol_desc = MolFromInchi(chemical_identifier, treatWarningAsError = False)

        else:
            mol_desc = MolFromSmiles(chemical_identifier)
        
        nBits = GetMorganFingerprintAsBitVect(mol = mol_desc, radius = radius, nBits = nBits, useChirality = useChirality)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(nBits, arr)

        nBitsToken = np.flatnonzero(arr > 0)
        del arr
    
    except:
        nBitsToken = None

    return nBitsToken



def RDKitMorganFingerprintExtractor(ChemicalIdentifiers: Tuple[str, str],
                                    radius: int,
                                    nBits: int,
                                    useChirality: bool) -> Tuple:
    
    RDKitnBitsTokens = None
    
    if ChemicalIdentifiers is not None:
        SMILES = ChemicalIdentifiers[0]
        if SMILES:
            RDKitnBitsTokens = GetRDKitMorganFingerprintTokens(SMILES, radius, nBits, useChirality)
                    
        if RDKitnBitsTokens is None:
            InChI = ChemicalIdentifiers[1]
            if InChI:
                RDKitnBitsTokens = GetRDKitMorganFingerprintTokens(InChI, radius, nBits, useChirality)

    return RDKitnBitsTokens



from rdkit.Chem import MACCSkeys

def GetRDKitMACCSkeysFingerprintTokens(chemical_identifier) -> np.ndarray:
    
    try:
        chemical_identifier = re.sub(invalidCharacters, "", chemical_identifier)
        
        if InChI_string_pattern.search(chemical_identifier):
            mol_desc = MolFromInchi(chemical_identifier, treatWarningAsError = False)

        else:
            mol_desc = MolFromSmiles(chemical_identifier)
        
        nBits = MACCSkeys.GenMACCSKeys(mol = mol_desc)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(nBits, arr)

        nBitsToken = np.flatnonzero(arr > 0)
        del arr
    
    except:
        nBitsToken = None

    return nBitsToken



def RDKitMACCSkeysFingerprintExtractor(ChemicalIdentifiers: Tuple[str, str]) -> Tuple:
    
    RDKitnBitsTokens = None
    
    if ChemicalIdentifiers is not None:
        SMILES = ChemicalIdentifiers[0]
        if SMILES:
            RDKitnBitsTokens = GetRDKitMACCSkeysFingerprintTokens(SMILES)
                    
        if RDKitnBitsTokens is None:
            InChI = ChemicalIdentifiers[1]
            if InChI:
                RDKitnBitsTokens = GetRDKitMACCSkeysFingerprintTokens(InChI)

    return RDKitnBitsTokens



def MINT_fingerprint_descriptor(chemical_identifiers, fingerprint_parameters):
    
    nBitsToken = None

    fingerprint_type = fingerprint_parameters[0]

    if isinstance(fingerprint_type, str):

        if fingerprint_type.__eq__('MACCSKeys'):
            ## ChemicalIdentifiers = (SMILES, InChI)
            ChemicalIdentifiers = (chemical_identifiers[0], chemical_identifiers[1])
            nBitsToken = RDKitMACCSkeysFingerprintExtractor(ChemicalIdentifiers = ChemicalIdentifiers)

        elif fingerprint_type.__eq__('MorganFingerprint'):
            ## ChemicalIdentifiers = (SMILES, InChI)
            ChemicalIdentifiers = (chemical_identifiers[0], chemical_identifiers[1])
            radius = fingerprint_parameters[2]
            nBits = fingerprint_parameters[3]
            useChirality = fingerprint_parameters[4]
            nBitsToken = RDKitMorganFingerprintExtractor(ChemicalIdentifiers = ChemicalIdentifiers, radius = radius, nBits = nBits, useChirality = useChirality)

        else:
            nBitsToken = [int(s) for s in chemical_identifiers[2].split("-")]
            nBitsToken = np.array(nBitsToken)
    
    return nBitsToken