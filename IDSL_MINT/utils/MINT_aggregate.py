import math

def MINT_aggregate(ID, MINTvec):
    """
    Aggregate a list of IDs by their corresponding vector of values.

    Args:
        ID (list): A list of IDs to be aggregated.
        MINTvec (list): A list of vector of values corresponding to the IDs.

    Returns:
        dict: A dictionary where each key is a unique MINT value found in MINTvec,
        and the corresponding value is a list of all IDs in ID that have that MINT vector value.

    Example:
        >>> ID = [1, 2, 3, 4, 5, 6]
        >>> MINTvec = [10, 20, 30, 10, 20, 40]
        >>> MINT_aggregate(ID, MINTvec)
        {10: [1, 4], 20: [2, 5], 30: [3], 40: [6]}
    """
    
    listMINT = {}
    for i in range(len(MINTvec)):
        if MINTvec[i] in listMINT:
            listMINT[MINTvec[i]].append(ID[i])
        else:
            listMINT[MINTvec[i]] = [ID[i]]
        
    return listMINT


def MINT_peak_aggregate(mspTrainingSet, max_number_ions_per_batch):
    
    ID, NumPeaksStr = [], []
    for i in range(len(mspTrainingSet)):
        if mspTrainingSet[i] is not None:
            ID.append(i)
            NumPeaksStr.append(mspTrainingSet[i][1])

    
    nPeaksDict = MINT_aggregate(ID, NumPeaksStr)

    nPeaksDictList = list(nPeaksDict.keys())

    msp_block_indices = []
    for NumPeaks in nPeaksDictList:
        indexMSPblock = nPeaksDict[NumPeaks]
        n = math.floor(max_number_ions_per_batch/NumPeaks)
        if n == 0:
            n = 1
        for i in range(0, len(indexMSPblock), n):
            msp_block_indices.append(indexMSPblock[i:i + n])

    return msp_block_indices