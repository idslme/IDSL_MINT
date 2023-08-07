import numpy as np

def spectral_entropy_calculator(spectra, allowedWeightedSpectralEntropy = True):

    spectra = np.array(spectra).reshape(-1, 2)

    spectra[:, 1] = spectra[:, 1]/np.sum(spectra[:, 1])

    spectral_entropy = -np.sum(spectra[:, 1] * np.log(spectra[:, 1]))

    if allowedWeightedSpectralEntropy:
        
        if spectral_entropy < 3:

            w = 0.25 + spectral_entropy*0.25
            spectra[:, 1] = np.power(spectra[:, 1], w)

            spectra[:, 1] = spectra[:, 1]/np.sum(spectra[:, 1])

            spectral_entropy = -np.sum(spectra[:, 1] * np.log(spectra[:, 1]))

        
    return spectral_entropy, spectra



def spectral_entropy_similarity_score(spectra_A, spectra_B, allowedWeightedSpectralEntropy = True, massError = 0):

    spectra_A = np.array(spectra_A).reshape(-1, 2)
    spectra_B = np.array(spectra_B).reshape(-1, 2)

    S_A, spectra_A = spectral_entropy_calculator(spectra_A, allowedWeightedSpectralEntropy)
    S_B, spectra_B = spectral_entropy_calculator(spectra_B, allowedWeightedSpectralEntropy)

    spectra_AB = spectra_AB_mixer(spectra_A, spectra_B, massError = massError)
    S_AB, _ = spectral_entropy_calculator(spectra_AB, allowedWeightedSpectralEntropy = False)

    entropy_similarity = 1 - (2*S_AB - (S_A + S_B))/1.38629436111989  ## log(4) = 1.38629436111989

    if entropy_similarity is None:
        entropy_similarity = 0
    
    elif entropy_similarity < 0:
        entropy_similarity = 0
    
    elif entropy_similarity > 1 and entropy_similarity < 2:
        entropy_similarity = 1 - entropy_similarity%1

    elif entropy_similarity >= 2:
        entropy_similarity = 0
    

    return entropy_similarity



def spectra_AB_mixer(spectra_A, spectra_B, massError = 0):
    
    nA = spectra_A.shape[0]
    nB = spectra_B.shape[0]

    spectra_A = np.concatenate((spectra_A, np.expand_dims(np.repeat(1, nA), axis = 1)), axis = 1)
    spectra_B = np.concatenate((spectra_B, np.expand_dims(np.repeat(2, nB), axis = 1)), axis = 1)
    nAB = nA + nB


    stacked_AB = np.concatenate((spectra_A, spectra_B), axis = 0)
    stacked_AB = stacked_AB[np.argsort(stacked_AB[:, 1], kind='mergesort')[::-1], :]
    

    spectra_AB = np.zeros(shape = (nAB, 2))
    counter_AB = 0


    for i in range(nAB):

        if stacked_AB[i, 0] > 0:
            
            x = np.flatnonzero((abs(stacked_AB[i, 0] - stacked_AB[:, 0]) <= massError) & (stacked_AB[i, 2] != stacked_AB[:, 2]))

            nx = len(x)

            if nx == 0:

                spectra_AB[counter_AB, 0] = stacked_AB[i, 0]
                spectra_AB[counter_AB, 1] = stacked_AB[i, 1]

                x = i
            
            else:

                if nx > 1:

                    xMin = np.argmin(abs(stacked_AB[i, 0] - stacked_AB[x, 0]))
                    x = np.expand_dims(x[xMin], axis = 0)

                x = np.concatenate((np.expand_dims(i, axis = 0), x), axis = 0)

                INT = np.sum(stacked_AB[x, 1])
                MZ = np.sum(stacked_AB[x, 0]*stacked_AB[x, 1])/INT

                spectra_AB[counter_AB, 0] = MZ
                spectra_AB[counter_AB, 1] = INT
            
            
            counter_AB += 1
            
            stacked_AB[x, :] = 0


    spectra_AB = spectra_AB[:counter_AB, :]

    return spectra_AB