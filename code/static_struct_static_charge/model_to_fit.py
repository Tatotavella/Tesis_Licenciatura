import sys
sys.path.append('../')
import ph_module as phm
import numpy as np
import pandas as pd

def constant_ph_model(data, Kelec, Ap, Anp, NeighMax, slope):

    matrix_for_fit = data

    # Self Penalty 
    polar = matrix_for_fit['Npi'].apply(lambda x: 1.0
                                        if x >= NeighMax[0]
                                        else np.exp(-slope[0]*(x-NeighMax[0])*(x-NeighMax[0])))
    
    non_polar = matrix_for_fit['Nnp'].apply(lambda x: 1.0
                                            if x >= NeighMax[1]
                                            else np.exp(-slope[1]*(x-NeighMax[1])*(x-NeighMax[1])))
    
    data = pd.DataFrame({'Elec': matrix_for_fit['Elec'],
                         'Up': polar,
                         'Unp': non_polar,
                         'Let': matrix_for_fit['Let'],
                         'pKaWat': matrix_for_fit['pKaWat']})

    # Correction for acids and bases in the direction of the Polarity Penalty
    data.loc[(data['Let'] == 'R') |
             (data['Let'] == 'K') |
             (data['Let'] == 'H'), 'Unp'] *= -1
    
    data.loc[(data['Let'] == 'E') |
             (data['Let'] == 'D'), 'Up'] *= -1

    # Polarity Interaction
    # Asp, Glu, Lys-Arg, His
    data.loc[(data['Let'] == 'D'), 'Up'] *= Ap[0]
    data.loc[(data['Let'] == 'E'), 'Up'] *= Ap[1]
    data.loc[(data['Let'] == 'K') |
             (data['Let'] == 'R'), 'Up'] *= Ap[2]
    data.loc[(data['Let'] == 'H'), 'Up'] *= Ap[3]

    data.loc[(data['Let'] == 'D'), 'Unp'] *= Anp[0]
    data.loc[(data['Let'] == 'E'), 'Unp'] *= Anp[1]
    data.loc[(data['Let'] == 'K') |
             (data['Let'] == 'R'), 'Unp'] *= Anp[2]
    data.loc[(data['Let'] == 'H'), 'Unp'] *= Anp[3]
    
    # Elec Interaction
    data['Elec'] *= Kelec

    # Computational Shift and pKa
    pka_comp = data['pKaWat'] + data['Elec'] + data['Unp'] + data['Up']

    return pka_comp
