#!/usr/bin/python3.5

#####################################################################
#
# Python Module with several functions to calculate 
# protein features involving pH effects
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

import numpy as np
import pandas as pd
from Bio.PDB import *

##
# List: Three letter code for aminoacids
##

res_list_three = ['GLY', 'ALA', 'VAL', 'LEU', 'MET',
                  'ILE', 'SER', 'THR', 'CYS', 'ASN',
                  'GLN', 'LYS', 'ARG', 'ASP', 'GLU',
                  'HIS', 'PHE', 'TYR', 'TRP', 'PRO']

##
# Aminoacids called "rings", for use in ring true positions calculations
# Three letter code is used. Also atoms in each ring and a dict to easy access.
##

ring_list_three = ['HIS', 'PHE', 'TYR', 'TRP']
HIS_rg_atoms = ['CG', 'ND1', 'CD2', 'CE1', 'NE2']
PHE_rg_atoms = ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']
TYR_rg_atoms = ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH']
TRP_rg_atoms = ['CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2']
rg_dict = {'HIS': HIS_rg_atoms,
           'PHE' : PHE_rg_atoms,
           'TYR' : TYR_rg_atoms,
           'TRP' : TRP_rg_atoms}

##
# Dictionary: Translate from three letter code to one letter code
##

three_to_one = {'ALA' : 'A', 'ARG' : 'R', 'ASN' : 'N', 'ASP' : 'D',
                'CYS' : 'C', 'GLU' : 'E', 'GLN' : 'Q', 'GLY' : 'G',
                'HIS' : 'H', 'ILE' : 'I', 'LEU' : 'L', 'LYS' : 'K',
                'MET' : 'M', 'PHE' : 'F', 'PRO' : 'P', 'SER' : 'S',
                'THR' : 'T', 'TRP' : 'T', 'TYR' : 'Y', 'VAL' : 'V'}

##
# Dictionary: Translate from three letter code to one letter type code of aminoacids
#
# Types: N -> Non Polar
#        P -> Polar
#        A -> Acid
#        B -> Base
#        G -> Glycine
##

type_dict = {'ALA' : 'N', 'ARG' : 'B', 'ASN' : 'P', 'ASP' : 'A', 'CYS' : 'P',
             'GLU' : 'A', 'GLN' : 'P', 'GLY' : 'G', 'HIS' : 'B', 'ILE' : 'N',
             'LEU' : 'N', 'LYS' : 'B', 'MET' : 'N', 'PHE' : 'N', 'PRO' : 'N',
             'SER' : 'P', 'THR' : 'P', 'TRP' : 'N', 'TYR' : 'P', 'VAL' : 'N'}

##
# Dictionary: Reference pKas in water for three letter code
##

ref_pkas = {'ASP' : 4.0, 'GLU' : 4.5, 'LYS' : 10.6, 'ARG' : 12.0, 'HIS' : 6.4}

##
# Dictionary: Reference pKas in water for one letter code
##

ref_pkas_one = {'D' : 4.0, 'E' : 4.5, 'K' : 10.6, 'R' : 12.0, 'H' : 6.4}

##
# Dictionary: Charged residues type in one letter code
#
# Types: A -> Acid
#        B -> Base
##

charged_classification = {'D': 'A', 'E': 'A', 'K': 'B',
                          'R': 'B', 'H': 'B', 'X': 'B', 'Z': 'A'}

##
# Three letter to one aminoacid code converter
##

def three_code_one(three_key):
    val = three_to_one[three_key]
    return val

##
# One letter charged aa to type
# Acid: A or Base: B result 
##

def charged_to_type(one_key):
    val = charged_classification[one_key]
    return val

##
# Function to assign reference pKas that are listed
# in the dictionary ref_pkas
##

def reference_pka(three_key):
    val = ref_pkas[three_key]
    return val

##
# Function to assign reference pKas that are listed
# in the dictionary ref_pkas_one FOR ONE LETTER CODE
##

def one_reference_pka(one_key):
    val = ref_pkas_one[one_key]
    return val

##
# Function to clasify aminoacids in types
# Types are: Acid, Base, Polar, NonPolar, Glycine
# Code type: A, B, P, N, G
# The input is the aminoacid in three letter code
# The output is the code letter.
##

def type_aa(three_key):
    val = type_dict[three_key]
    return val

##
# Function to clean a list of elements from a PDB and get only aminoacids.
# It cleans the PDB from other molecules that are not aminoacids. 
# Recieves the full list of residues as input and returns a list only with
# aminoacids. It uses the list defined above called res_list_three.
##

def only_residues(full_list):
    output_res = []
    for r in full_list:
        r_name = r.get_resname()
        if r_name in res_list_three:
            output_res.append(r)
    return output_res

##
# Functions to calculate a geometric "true position" for aminoacid with rings.
# Receives the residue and based on identity returns a "true position" (array tp)
# for the ring.
##

def ring_position(ring_residue):
    rg_name = ring_residue.get_resname()
    if rg_name not in ring_list_three:
        raise ValueError('The residue is not in rings list')
    at_in_rg = rg_dict[rg_name]
    # List of relevant atoms for the ring
    atom_values = []
    for a in ring_residue:
        atom_name = a.get_name()
        if atom_name in at_in_rg:
            atom_values.append(a)
    # Geometric center calculation
    norm = len(atom_values)
    x_mean = 0.0
    y_mean = 0.0
    z_mean = 0.0

    for a in atom_values:
        pos = a.get_vector()
        x_mean += pos[0]
        y_mean += pos[1]
        z_mean += pos[2]


    x_mean = x_mean/norm
    y_mean = y_mean/norm
    z_mean = z_mean/norm

    true_pos = [x_mean, y_mean, z_mean]

    return true_pos

##
# Function to clean PDB and get important data
# for ca cb atoms.
# - Input: pdb id, struct num, chain name
# - Output: ca posit, rep posit, letters, types
#           As dicionary with keys: ca, cb, let, typ
##

def clean_pdb(pdb_id, struct_num, chain_name, repre):

    # Calculate for ca and side chain representation
    # Representations availiable = ['ca', 'cb', 'long']
    representations = ['ca', repre]

    # Read PDB files
    parser = PDBParser()
    structure = parser.get_structure('-','../../PDBs/'+pdb_id+'.pdb')
    all_chains = structure[struct_num]
    chain = all_chains[chain_name]
    model = chain
    # List of residues and other molecules
    full_list = model.get_residues()
    # Get only a list of residues
    only_r = only_residues(full_list)

    
    # Positions for residue (side chain) and backbone
    # pos = [(res1_x, res1_y, res1_z), ... , (... , resN_y, resN_z)]
    resi_pos = []
    back_pos = []

    # Array of numeration from pdb
    real_num = []

    # Letters and types of residues arrays
    lett_res = []
    type_res = []

    for rep in representations:
        # File that indicates where to place residues (point in space)
        # to calculate distances (representation)
        rep_file = 'rep_' + rep + '.txt'

        # Load and create Dictionary of representation.
        # Tells where each type of residue is placed
        res_ref, places = np.genfromtxt('../../files/'+rep_file,
                                        skip_header = 1,
                                        unpack = True,
                                        dtype = 'str')
        info_dict = {}
        for idx,r in enumerate(res_ref):
            info_dict[r] = places[idx]

        # Write position of residues and information
        for rdx, r in enumerate(only_r):
            name = r.get_resname()
            id_res = r.get_id()
            repres = info_dict[name]
            one_let = three_code_one(name)
            res_type = type_aa(name)
            key = id_res[1]
            if repres == 'RG':
                # The residue was labelled as ring.
                # The residue is placed at the ring's mean position
                # Ring mean position
                tp = ring_position(r)
                if rep == 'ca':
                    back_pos.append([tp[0],tp[1],tp[2]])
                    real_num.append(key)
                    lett_res.append(one_let)
                    type_res.append(res_type)
                else:
                    resi_pos.append([tp[0],tp[1],tp[2]])
            else:
                # Other residues and representations
                # Atom position, try representation,
                # if there is a problem place it in CB
                try:
                    atom_res = r[repres]
                except KeyError:
                    atom_res = r['CB']
                tp = atom_res.get_vector()
                if rep == 'ca':
                    back_pos.append([tp[0],tp[1],tp[2]])
                    real_num.append(key)
                    lett_res.append(one_let)
                    type_res.append(res_type)
                else:
                    resi_pos.append([tp[0],tp[1],tp[2]])

    # Terminals position: N-ter is called 'X', C-ter is callez 'Z'

    # N-ter
    r = only_r[0]
    name = r.get_resname()
    id_res = r.get_id()
    repres = info_dict[name]
    one_let = three_code_one(name)
    res_type = type_aa(name)
    key = id_res[1]
    atom_res = r['N']
    tp = atom_res.get_vector()
    real_num.append(len(only_r) + 1)
    lett_res.append('X')
    type_res.append('X')
    back_pos.append([tp[0],tp[1],tp[2]])
    resi_pos.append([tp[0],tp[1],tp[2]])

    # C-ter
    r = only_r[-1]
    name = r.get_resname()
    id_res = r.get_id()
    repres = info_dict[name]
    one_let = three_code_one(name)
    res_type = type_aa(name)
    key = id_res[1]
    atom_res = r['C']
    tp = atom_res.get_vector()
    real_num.append(len(only_r) + 2)
    lett_res.append('Z')
    type_res.append('Z')
    back_pos.append([tp[0],tp[1],tp[2]])
    resi_pos.append([tp[0],tp[1],tp[2]])
    
    # Output as dictionary
    out = {}
    out['nr'] = real_num
    out['ca'] = back_pos
    out['pos'] = resi_pos
    out['let'] = lett_res
    out['typ'] = type_res
    return out

##
# Function to calculate delta self for exhaustive parameter exploration
##

def self_delta(Np,Nnp,NpMx,NnpMx,alfp,alfnp):
    term = 0.0
    u_p = 0.0
    u_np = 0.0
    # Polar
    if Np < NpMx:
        u_p = np.exp(-alfp*(Np-NpMx)*(Np-NpMx))
    elif Np >= NpMx:
        u_p = 1.0

    # Non Polar
    if Nnp < NnpMx:
        u_np = np.exp(-alfnp*(Nnp-NnpMx)*(Nnp-NnpMx))
    elif Nnp >= NnpMx:
        u_np = 1.0

    term = [- u_p, u_np]
    return term

##
# Function to calculate electrostatic contribution with static charges
# from distance matrix
#
# Input:
# - PDB id
# - Representation of positions: ca, cb, long
# - Screening distance: l
# - Ionizable indexes starting from 0
# - Charge of each residue
# Output:
# - DataFrame with electrostatic contribution
##

def electro_from_matrix(pdb_id,rep,l,charged_indexes,charges):

    matrix_dir = '../../processed/'
    matrix_f_name = 'matrix_pos_' + rep + '_' + pdb_id + '.txt'

    # Import distance matrix
    dataset = pd.read_csv(matrix_dir + matrix_f_name,
                          sep = '\t', header = None)

    # Charged Submatrixq
    charged_submatrix = dataset.loc[charged_indexes,charged_indexes]
    electro_field = np.exp(-charged_submatrix/l)/charged_submatrix
    electro_charged = electro_field.mul(charges, axis = 1)
    electro_penalty = electro_charged.replace([np.inf, -np.inf], 0)
    elec = pd.DataFrame({'Elec': electro_penalty.sum(axis = 1)})

    return elec

##
# Function to calculate three types of neighbours from distance matrix.
# Neighbour types are: Polar, Non-polar, Ionizable
#
# Input:
# - PDB id
# - Representation of positions: ca, cb, long
# - Counting Slope in 3-array shape, first index for polar,
#                                    second for non polar,
#                                    third for ionizable
# - Maximum Neighbour Cutoff in 3-array shape, first index for polar,
#                                              second for non polar,
#                                              third for ionizable
#
# Output:
# - Pandas DataFrame with Neighbour information.
#          Columns named Np, Nnp, Ni for each type of neighbour.
#          There is a column for the sum of Np and Ni called Npi.
##

def neighbours_from_matrix(pdb_id,rep,charged_indexes,alpha,r_max):

    matrix_dir = '../../processed/'
    matrix_f_name = 'matrix_pos_' + rep + '_' + pdb_id + '.txt'
    posit_dir = '../../processed/'
    posit_f_name = 'info_positions_' + rep + '_' + pdb_id + '.txt' 

    # Import distance matrix and position dataset
    matrix_dataset = pd.read_csv(matrix_dir + matrix_f_name,
                                 sep = '\t', header = None)
    
    position_dataset = pd.read_csv(posit_dir + posit_f_name,
                                   sep = '\t')
    
    polar_indexes = position_dataset.loc[position_dataset['Type'] == 'P'].index
    non_polar_indexes = position_dataset.loc[position_dataset['Type'] == 'N'].index
    
    # Charged Submatrix
    charged_polar_submatrix = matrix_dataset.loc[charged_indexes,
                                                 polar_indexes]
    
    charged_non_polar_submatrix = matrix_dataset.loc[charged_indexes,
                                                     non_polar_indexes]
    
    charged_charged_submatrix = matrix_dataset.loc[charged_indexes,
                                                   charged_indexes]

    # Neighbour Counting
    charged_polar = charged_polar_submatrix.applymap(
                    lambda x: 1.0 if x <= r_max[0]
                    else np.exp(-alpha[0]*(x-r_max[0])*(x-r_max[0])))

    charged_non_polar = charged_non_polar_submatrix.applymap(
                        lambda x: 1.0 if x <= r_max[1]
                        else np.exp(-alpha[1]*(x-r_max[1])*(x-r_max[1])))
    
    charged_charged = charged_charged_submatrix.applymap(
                      lambda x: 0.0 if x == 0.0
                      else (1.0 if x <= r_max[2]
                      else np.exp(-alpha[2]*(x-r_max[2])*(x-r_max[2]))))

    # Total Neighbour Number
    polar = charged_polar.sum(axis = 1)
    non_polar = charged_non_polar.sum(axis = 1)
    ionizable = charged_charged.sum(axis = 1)
    
    dataset = pd.DataFrame({'Np': polar , 'Nnp': non_polar,
                            'Ni': ionizable, 'Npi': polar + ionizable,
                            'Let': position_dataset.loc[charged_indexes,'Let']})

    return dataset

##
# Function to calculate polarity penalty with neighbours
# calculated from neighbours_from_matrix() function.
#
# Input:
# - Neighbours DataFrame which comes from the output of neighbours_from_matrix()
# - Slope values for penalty in a 2-array shape, first index for polar,
#                                                second index for non polar
# - Maximum Neighbour Cutoff values for penalty in a 2-array shape,
#                                              first index for polar,
#                                              second index for non polar
# - Polar Penalty Strength in a 3-array shape, first index for Glu-Asp,
#                                              second index for Lys-Arg,
#                                              third index for His.
# - Non Polar Penalty Strength in a 3-array shape, first index for Glu-Asp,
#                                                  second index for Lys-Arg,
#                                                  third index for His.
#
# Output:
# - Pandas DataFrame with penalty information.
#   Columns named Polar, Non-Polar, Let
#   for each type of penalty and letter of residue
##

def polarity_from_neighbours(neighbours, slope, NeighMax, Ap, Anp):
    # Calculation
    p_pen = neighbours['Npi'].apply(lambda x: 1.0 if x >= NeighMax[0]
                                    else np.exp(-slope[0]*
                                                (x-NeighMax[0])*
                                                (x-NeighMax[0])))
    
    n_p_pen = neighbours['Nnp'].apply(lambda x: 1.0 if x >= NeighMax[1]
                                      else np.exp(-slope[1]*
                                                  (x-NeighMax[1])*
                                                  (x-NeighMax[1])))

    # Output DataFrame
    self_penalty = pd.DataFrame({'Polar': p_pen, 'Non-Polar': n_p_pen,
                                 'Let': neighbours['Let']})

    # Signs for acids and bases 
    self_penalty.loc[(self_penalty['Let'] == 'R') |
                     (self_penalty['Let'] == 'K') |
                     (self_penalty['Let'] == 'H') |
                     (self_penalty['Let'] == 'X'),'Non-Polar'] *= -1
    
    self_penalty.loc[(self_penalty['Let'] == 'E') |
                     (self_penalty['Let'] == 'D') |
                     (self_penalty['Let'] == 'Z'),'Polar'] *= -1

    return self_penalty
