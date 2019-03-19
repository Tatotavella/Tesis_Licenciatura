#!/usr/bin/python3.5

#####################################################################
#
# Python Script that receives a PDB file and returns two files: One 
# of them encodes the position of each amino-acid, its type and
# one-letter amino-acid code. The other one is a distance matrix
# between all residues. The PDB id is received as input and the
# file is loaded from the /PDBs directory. The output files are
# saved in the /processed directory.
#
# NOTE: Structure number and chain name from PDB are hardcoded
#       to be 0 and A respectively. Also Beta-Carbon position
#       of side-chain is hardcoded in variable rep. The core
#       function which does the calculation is clean_pdb() from
#       the ph_module.
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
from scipy.spatial import distance

##
# pH Module
##
import ph_module as phm

##
# Error handling in input
##
if len(sys.argv) <= 1:
    print('Error. PDB id should be entered as input for script. Usage:')
    print('python clean_pdb.py PDBid')
else:
    pdb_id = sys.argv[1]

##    
# Structure Number and Chain Name
##
struct_num = 0
chain_name = 'A'

##
# Position of side-chain
##
rep = 'cb'

##
# Get Ca Pos Let Type data from PDB
##
out = phm.clean_pdb(pdb_id, struct_num, chain_name, rep)

##
# Convert to Pandas DataFrame
##
ca = np.array(out['ca'])
cax = ca[:,0]
cay = ca[:,1]
caz = ca[:,2]

pos = np.array(out['pos'])
posx = pos[:,0]
posy = pos[:,1]
posz = pos[:,2]

let = np.array(out['let'])
typ = np.array(out['typ'])
nr = np.array(out['nr'])
nidx = np.arange(1,len(nr)+1)

dataset = pd.DataFrame({'Nr': nr, 'Nidx': nidx, 'CaX': cax,
                        'CaY': cay, 'CaZ': caz, 'PosX': posx,
                        'PosY': posy, 'PosZ': posz, 'Let': let, 'Type': typ})
header = ['Nr','Nidx','CaX','CaY','CaZ','PosX','PosY','PosZ','Let','Type']
dataset = dataset[header]

##
# Matrix of euclidean distances
##
dist_matrix = distance.cdist(pos,pos, 'euclidean')
matrix_dataset = pd.DataFrame(data = dist_matrix, columns = nr)

##
# Output file info_positions_rep_PDBid.txt
##
posit_direc = '../../processed/'
posit_f_name = 'info_positions_' + rep + '_' + pdb_id + '.txt'
dataset.to_csv(posit_direc + posit_f_name, sep='\t', index = False)
print('Positions saved in: ' + posit_direc + posit_f_name)

##
# Output file matrix_rep_PDBid.txt
##
matrix_direc = '../../processed/'
matrix_f_name = 'matrix_pos_' + rep + '_' + pdb_id + '.txt'
matrix_dataset.to_csv(matrix_direc + matrix_f_name, sep='\t',
                      index = False, header = False)
print('Distance Matrix saved in: ' + matrix_direc + matrix_f_name)
