#!/usr/bin/python3.5

#####################################################################
#
# Python Module with several functions to calculate 
# stochastic constant pH simulations. The Metropolis-Hastings
# algoithm is implemented in the charge state sampling.
#
# Authors: Franco Tavella & Ernesto Roman 
# Contact: tavellafran@gmail.com
# Institution: Universidad de Buenos Aires, Argentina
#
# Dependencies: - NumPy 1.16.0   (Python 3.5 library)
#               - Pandas 0.23.4  (Python 3.5 library)
#               - Biopython 1.73 (Python 3.5 library)
#
####################################################################

import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import random

##
# Classification of charge residues
##
charged_classification = {'D': 'A', 'E': 'A', 'K': 'B',
                          'R': 'B', 'H': 'B','X':'B','Z':'A'}

##
# Charged residue one-letter amino-acid code to classification
##
def charged_to_type(one_key):
    val = charged_classification[one_key]
    return val

##
# Single charge flip based on acid or base behaviour.
# Acids switch between 0 and -1. Bases switch between
# 0 and 1. Type in encoded in input variable type_aob
##
def charge_flip(type_aob, old_charge):
    new_charge = 0.0
    if type_aob == 'A':
        if old_charge == -1.0:
            new_charge = 0.0
        elif old_charge == 0.0:
            new_charge = -1.0
    if type_aob == 'B':
        if old_charge == 1.0:
            new_charge = 0.0
        elif old_charge == 0.0:
            new_charge = 1.0
    return new_charge

##
# Contribution due to reference pKa
##
def ph_mc(ph, pkaw):
    term = ph - pkaw
    return term

##
# Contribution due to electrostatic interactions based
# on current charge state
##
def single_electro_mc(charge_state, charged_submatrix, place, l):
    
    electro_field = np.exp(-charged_submatrix/l)/charged_submatrix
    electro_charged = electro_field.mul(charge_state['State'],
                                        axis = 1)
    electro_penalty = electro_charged.replace([np.inf, -np.inf], 0)
    to_sum = electro_penalty.loc[place]
    elec = to_sum.sum(axis = 0)

    return elec

##
# Total electrostatic energy based on current charge
# state
##
def full_electro_energy(charge_state, charged_submatrix, l):
    
    electro_field = np.exp(-charged_submatrix/l)/charged_submatrix
    electro_charged_aux = electro_field.mul(charge_state['State'],
                                            axis = 1)
    electro_charged = electro_charged_aux.mul(charge_state['State'],
                                              axis = 0)
    electro_penalty = electro_charged.replace([np.inf, -np.inf], 0)
    electro_sum = electro_penalty.sum(axis = 1)
    
    # Divide by two to get correct value
    elec_energy= electro_sum.sum(axis = 0) / 2

    return elec_energy

##
# Main loop of Metropolis-Hsatings algorithm to sample charge state
##
def mc_loop(ph, N_steps, charge_state, charged_submatrix,
            self_penalty, K_elec_T, l):
    
    charged_indexes = np.array(self_penalty.index)

    history_states = pd.DataFrame({'Init':charge_state['State']},
                                  index = charged_indexes)
    history_elec_energy = np.zeros(N_steps)

    ##########################
    # Start of the MC Loop
    ##########################
    for n in range(N_steps):
        # Select random titratable residue
        place = random.choice(charged_indexes)

        # Set new and old charge
        old_charge = charge_state.loc[place,'State']
        type_aob = charged_to_type(self_penalty.loc[place,'Let'])
        new_charge = charge_flip(type_aob, old_charge)

        # Direction and qab
        direction = new_charge - old_charge

        # pKa Water contribution
        pkaw = self_penalty.loc[place,'pKaWat']
        delta_ph = ph_mc(ph, pkaw)
        
        # Calculate Elec energy centered in that selected residue
        delta_elec = single_electro_mc(charge_state, charged_submatrix,
                                       place, l)

        # Polarity Contribution
        delta_self = -1 * (self_penalty.loc[place,'Non-Polar'] +
                           self_penalty.loc[place,'Polar'] )

        # Free energy Difference Between States
        delta_mc = direction * np.log(10) * (delta_ph +
                                             K_elec_T * delta_elec +
                                             delta_self)

        if delta_mc < 0:
            charge_state.loc[place,'State'] = new_charge
        elif delta_mc >= 0:
            random_prob = random.uniform(0,1)
            if random_prob > np.exp(-delta_mc):
                # No Change
                charge_state.loc[place,'State'] = old_charge
            else:
                # Change
                charge_state.loc[place,'State'] = new_charge

        # Save Charge State
        history_states[n] = charge_state['State']
        # Save Electrostatic Energy
        energy_elec = full_electro_energy(charge_state,
                                          charged_submatrix, l)
        history_elec_energy[n] = K_elec_T * energy_elec
        
    ######################
    # End of the MC Loop
    ######################
    return [history_states, history_elec_energy]

