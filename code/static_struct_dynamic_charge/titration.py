import sys
sys.path.append('../')
import ph_module as phm
import mc_module as mcm
import numpy as np
import pandas as pd
from datetime import datetime

##########
# Inputs
##########
pdb_id     = str(sys.argv[1])
ph         = float(sys.argv[2])
Kelec      = float(sys.argv[3])
repetition = str(sys.argv[4])

#################
# Script Timing
#################
startTime = datetime.now()

#######################
# Initial Parameters
#######################
k_b = 0.0019872041 # kcal/(mol*K)
T   = 300          # K
l   = 10.0         # Angstroms
alpha = [0.1, 0.1, 0.1] # [Polar, Non-Polar, Ionizable]
r_max = [5.0, 7.0, 5.0] # [Polar, Non-Polar, Ionizable]
Ap = [0.749, 0.429, 0.112, 0.096] # [Asp-Glu, Lys-Arg, His]
Anp = [3.836, 4.107, 5.628, 4.924] # [Asp-Glu, Lys-Arg, His]
pka_wat = [4.0, 4.5, 10.6, 12.0, 6.4, 8.0, 3.6] # [Asp, Glu, Lys, Arg, His, N-ter, C-ter]
NeighMax = [4.580, 12.741] # [Polar, Non-Polar]
slope = [0.1, 10] # [Polar, Non-Polar]

########################
# Import position data
########################
representation = 'cb'
dataset = pd.read_csv('../../processed/info_positions_'
                      + representation + '_' + pdb_id + '.txt', sep = '\t')
charged = dataset.loc[(dataset['Type'] == 'A') |
                      (dataset['Type'] == 'B') |
                      (dataset['Type'] == 'X') |
                      (dataset['Type'] == 'Z')]
charged_indexes = np.array(charged.index)
data_matrix = pd.read_csv('../../processed/matrix_pos_'
                          + representation + '_' + pdb_id
                          + '.txt', sep = '\t', header = None)

charged_submatrix = data_matrix.loc[charged_indexes,charged_indexes]

############################################################
# Neighbour Counting and Polarity Environment Contribution
############################################################
neighbours = phm.neighbours_from_matrix(pdb_id, representation,
                                        charged_indexes, alpha, r_max)
self_penalty = phm.polarity_from_neighbours(neighbours, slope, NeighMax, Ap, Anp)

# Strength of the interaction
self_penalty.loc[(self_penalty['Let'] == 'D'),'Polar'] *= Ap[0]
self_penalty.loc[(self_penalty['Let'] == 'E'),'Polar'] *= Ap[1]
self_penalty.loc[(self_penalty['Let'] == 'K') |
                 (self_penalty['Let'] == 'R'), 'Polar'] *= Ap[2]
self_penalty.loc[(self_penalty['Let'] == 'H'),'Polar'] *= Ap[3]

self_penalty.loc[(self_penalty['Let'] == 'D'),'Non-Polar'] *= Anp[0]
self_penalty.loc[(self_penalty['Let'] == 'E'),'Non-Polar'] *= Anp[1]
self_penalty.loc[(self_penalty['Let'] == 'K') |
                 (self_penalty['Let'] == 'R'), 'Non-Polar'] *= Anp[2]
self_penalty.loc[(self_penalty['Let'] == 'H'),'Non-Polar'] *= Anp[3]

#################
# Reference pKa
#################
self_penalty['pKaWat'] = np.zeros(len(self_penalty['Let']))
self_penalty.loc[self_penalty['Let'] == 'D','pKaWat'] = pka_wat[0]
self_penalty.loc[self_penalty['Let'] == 'E','pKaWat'] = pka_wat[1]
self_penalty.loc[self_penalty['Let'] == 'K','pKaWat'] = pka_wat[2]
self_penalty.loc[self_penalty['Let'] == 'R','pKaWat'] = pka_wat[3]
self_penalty.loc[self_penalty['Let'] == 'H','pKaWat'] = pka_wat[4]
self_penalty.loc[self_penalty['Let'] == 'X','pKaWat'] = pka_wat[5]
self_penalty.loc[self_penalty['Let'] == 'Z','pKaWat'] = pka_wat[6]

########################
# Energy normalization
########################
self_penalty['Polar'] *= 1/(k_b * T * np.log(10))
self_penalty['Non-Polar'] *= 1/(k_b * T * np.log(10))
K_elec_T = Kelec / (k_b * T * np.log(10))

#######################
# Ensemble properties
#######################
ensemble_size = 20000
equilib_time = 15000
decorr_time = 100
N_steps = ensemble_size * decorr_time + equilib_time

################
# Calculation
################
initial_state = np.array(charged['Type'].replace(['A','B','X','Z'],[-1.0,1.0,1.0,-1.0]))
charge_state = pd.DataFrame({'State': initial_state},index = self_penalty.index)
history_states, history_elec_energy = mcm.mc_loop(ph, N_steps,
                                                  charge_state,
                                                  charged_submatrix,
                                                  self_penalty, K_elec_T, l)
ensemble_states = history_states.iloc[:, equilib_time::decorr_time]
ensemble_states.drop(ensemble_states.columns[-1], axis = 1, inplace = True)
ensemble_energy = pd.DataFrame({ 'Elec': history_elec_energy[equilib_time::decorr_time]})
ensemble_states.to_csv('./titration_data/states_'
                       + pdb_id + '_' + str(ph) + '_'
                       + str(Kelec) + '_' + repetition + '.txt', sep = '\t')
ensemble_energy.to_csv('./titration_data/elec_'
                       + pdb_id + '_' + str(ph) + '_'
                       + str(Kelec) + '_' + repetition + '.txt', sep = '\t')
#################
# Script timing
#################
print("Time elapsed")
print(datetime.now() - startTime)
