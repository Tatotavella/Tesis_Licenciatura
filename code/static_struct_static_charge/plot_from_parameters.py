import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import scipy

# pH Module
import ph_module as phm

#######################################################################################################
#
# Parameters
#
#######################################################################################################

k_b = 0.0019872041 # kcal/(mol*K)
T = 300 # K
l = 10.0 # Angstroms

Kelec = 4.85

alpha = [0.1, 0.1, 0.1] # [Polar, Non-Polar, Ionizable]
r_max = [5.0, 7.0, 5.0] # [Polar, Non-Polar, Ionizable]
Ap = [0.749, 0.429, 0.112,  0.096] # [Asp, Glu, Lys-Arg, His]
Anp = [4.72, 4.2, 5.43, 4.87] # [Asp, Glu, Lys-Arg, His]
pka_wat = [4.0, 4.5, 10.6, 12.0, 6.4, 8.0, 3.6] # [Asp, Glu, Lys, Arg, His, N-ter, C-ter]

NeighMax = [4.580, 12.741] # [Polar, Non-Polar]
slope = [0.1, 2.46] # [Polar, Non-Polar]

#######################################################################################################
#
# DataFrame for Results
#
#######################################################################################################

pdb_list = np.genfromtxt('../../files/pdb_list.txt', dtype = str)
result = pd.read_csv('../../files/pka_exp.txt', sep = '\t')
datapoints = len(result['pKaExp'])

# Extend result DataFrame
result['pKaComp'] = np.zeros(datapoints)
result['ShiftComp'] = np.zeros(datapoints)
result['ShiftExp'] = np.zeros(datapoints)
result['Elec'] = np.zeros(datapoints)
result['Unp'] = np.zeros(datapoints)
result['Up'] = np.zeros(datapoints)
result['Nnp'] = np.zeros(datapoints)
result['Np'] = np.zeros(datapoints)
result['Ni'] = np.zeros(datapoints)
result['Npi'] = np.zeros(datapoints)
result['pKaWat'] = np.zeros(datapoints)
result['Let'] = np.zeros(datapoints)
rep = 'cb'

#######################################################################################################
#
# Calculation
#
#######################################################################################################

for pdb_id in pdb_list:
    dataset = pd.read_csv('../../processed/info_positions_' + rep + '_' + pdb_id + '.txt', sep = '\t')
    
    charged = dataset.loc[(dataset['Type'] == 'A') |
                          (dataset['Type'] == 'B') |
                          (dataset['Type'] == 'X') |
                          (dataset['Type'] == 'Z')]
    
    charged_indexes = np.array(charged.index)
    
    # Calculate Electrostatic Contribution
    charges = np.array(charged['Type'].replace(['A','B','X','Z'],[-1,1,1,-1]))
    elec = phm.electro_from_matrix(pdb_id,rep,l,charged_indexes,charges)

    # Count Neighbours
    neighbours = phm.neighbours_from_matrix(pdb_id,rep,charged_indexes,alpha,r_max)

    # Calculate Polarity Environment Contribution
    self_penalty = phm.polarity_from_neighbours(neighbours,slope,NeighMax,Ap,Anp)

    # Reference pKa
    ref_pka = pd.DataFrame({'pKaWat': np.zeros(len(self_penalty['Let'])),
                            'Let': self_penalty['Let']}, index = self_penalty.index)
    ref_pka.loc[ref_pka['Let'] == 'D','pKaWat'] = pka_wat[0]
    ref_pka.loc[ref_pka['Let'] == 'E','pKaWat'] = pka_wat[1]
    ref_pka.loc[ref_pka['Let'] == 'K','pKaWat'] = pka_wat[2]
    ref_pka.loc[ref_pka['Let'] == 'R','pKaWat'] = pka_wat[3]
    ref_pka.loc[ref_pka['Let'] == 'H','pKaWat'] = pka_wat[4]
    ref_pka.loc[ref_pka['Let'] == 'X','pKaWat'] = pka_wat[5]
    ref_pka.loc[ref_pka['Let'] == 'Z','pKaWat'] = pka_wat[6]
    
    # Correct indexes to fill results     
    pka_exp_idx = result.loc[result['PDB'] == pdb_id.upper(),'Nidx']

    # Note: Minus in Elec Penalty and Energy normalization with (k_b * T)
    new_column = pd.DataFrame({'Let': self_penalty.loc[pka_exp_idx,'Let'],
                               'Elec': elec.loc[pka_exp_idx,'Elec'] * (-1),
                               'Unp': self_penalty.loc[pka_exp_idx, 'Non-Polar'],
                               'Up': self_penalty.loc[pka_exp_idx, 'Polar'],
                               'pKaWat': ref_pka.loc[pka_exp_idx, 'pKaWat'],
                               'Np': neighbours.loc[pka_exp_idx,'Np'],
                               'Nnp': neighbours.loc[pka_exp_idx,'Nnp'],
                               'Ni': neighbours.loc[pka_exp_idx,'Ni'],
                               'Npi': neighbours.loc[pka_exp_idx,'Npi']})

    
    new_column = new_column.set_index(result.loc[result['PDB'] == pdb_id.upper()].index)

    # Update Results
    result.update(new_column)


# Experimental Shift
result['ShiftExp'] = result['pKaExp'] - result['pKaWat']

################
# Special Cases
################
special_columns = ['PDB','Nidx','pKaExp','ShiftExp','Let','Npi','Nnp']
special_cases = result.loc[(result['Nnp'] > 10) |
                           (result['ShiftExp'] > 1) |
                           (result['ShiftExp'] < -2),special_columns]

special_cases.to_csv('special_cases.txt', sep = '\t')


#######################################################################################################
#
# Strength of interactions
#
#######################################################################################################

# Electrostatic

result['Elec'] *= Kelec / (k_b * T * np.log(10))

# Asp, Glu, Lys-Arg, His

# Polar
result.loc[(result['Let'] == 'D'), 'Up'] *= Ap[0] / (k_b * T * np.log(10))
result.loc[(result['Let'] == 'E'), 'Up'] *= Ap[1] / (k_b * T * np.log(10))
result.loc[(result['Let'] == 'K') |
           (result['Let'] == 'R'), 'Up'] *= Ap[2] / (k_b * T * np.log(10))
result.loc[(result['Let'] == 'H'), 'Up'] *= Ap[3] / (k_b * T * np.log(10))

# Non-Polar
result.loc[(result['Let'] == 'D'), 'Unp'] *= Anp[0] / (k_b * T * np.log(10))
result.loc[(result['Let'] == 'E'), 'Unp'] *= Anp[1] / (k_b * T * np.log(10))
result.loc[(result['Let'] == 'K') |
           (result['Let'] == 'R'), 'Unp'] *= Anp[2] / (k_b * T * np.log(10))
result.loc[(result['Let'] == 'H'), 'Unp'] *= Anp[3] / (k_b * T * np.log(10))

    
# Calculate Shift and pKa
result['pKaComp'] = result['pKaWat'] + result['Elec'] + result['Unp'] + result['Up']
result['ShiftComp'] = result['Elec'] + result['Unp'] + result['Up']

# Save Result
result.to_csv('result_final.txt', sep = '\t')

# Error calculation
null_model = np.sqrt(np.sum(np.square(result['pKaExp'] - result['pKaWat']))/datapoints)
rmsd = np.sqrt(np.sum(np.square(result['pKaExp'] - result['pKaComp']))/datapoints)

print('RMSD: %.3f' % (rmsd))


#######################################################################################################
#
# Plot Data - Shift
#
#######################################################################################################

fig, ax1 = plt.subplots()

# Shift Plot
glu_s, = ax1.plot(result.loc[result['Let'] == 'E','ShiftExp'],
                  result.loc[result['Let'] == 'E','ShiftComp'],
                  linestyle = 'None', marker = 'o', color = 'b',
                  label = 'Glu', markeredgecolor = 'k', zorder = 3)

asp_s, = ax1.plot(result.loc[result['Let'] == 'D','ShiftExp'],
                  result.loc[result['Let'] == 'D','ShiftComp'],
                  linestyle = 'None', marker = '^', color = 'r',
                  label = 'Asp', markeredgecolor = 'k', zorder = 4)

his_s, = ax1.plot(result.loc[result['Let'] == 'H','ShiftExp'],
                  result.loc[result['Let'] == 'H','ShiftComp'],
                  linestyle = 'None', marker = 's', color = 'y',
                  label = 'His', markeredgecolor = 'k', zorder = 5)

lys_s, = ax1.plot(result.loc[result['Let'] == 'K','ShiftExp'],
                  result.loc[result['Let'] == 'K','ShiftComp'],
                  linestyle = 'None', marker = 'D', color = 'c',
                  label = 'Lys', markeredgecolor = 'k', zorder = 6)

ax1.grid(True, which = 'both')
ax1.plot([-5,5],[-5,5], linestyle = '-', color = 'black')
ax1.plot([-5,5],[0,0], linestyle = '-', color = 'black')
ax1.plot([0,0],[-5,5], linestyle = '-', color = 'black')
ax1.plot([-4,5],[-5,4], linestyle = '--', color = 'green')
ax1.plot([-5,4],[-4,5], linestyle = '--', color = 'green')

ax1.set_xlabel('Corrimiento Experimental', fontsize = 15)
ax1.set_ylabel('Corrimiento Calculado', fontsize = 15)

ax1.set_xticks(np.arange(-5, 6, step=1))
ax1.set_yticks(np.arange(-5, 6, step=1))
ax1.set_xlim([-5,5])
ax1.set_ylim([-5,5])

ax1.tick_params(axis = 'both', which = 'major', labelsize = 15)
ax1.set_aspect('equal')

legend_1 = ax1.legend(numpoints = 1,
                      fontsize = 15,
                      title = 'Tipo de Residuo',
                      loc = 'upper left')

plt.setp(legend_1.get_title(), fontsize = 15)
plt.tight_layout()
#'''

'''
#######################################################################################################
#
# Plot Data - pKa
#
#######################################################################################################

# pKaExp Plot
glu_p, = ax2.plot(result.loc[result['Let'] == 'E','pKaExp'], result.loc[result['Let'] == 'E','pKaComp'], linestyle = 'None', marker = 'o', color = 'b', label = 'Glu', markeredgecolor = 'k', zorder = 3)
asp_p, = ax2.plot(result.loc[result['Let'] == 'D','pKaExp'], result.loc[result['Let'] == 'D','pKaComp'], linestyle = 'None', marker = '^', color = 'r', label = 'Asp', markeredgecolor = 'k', zorder = 4)
his_p, = ax2.plot(result.loc[result['Let'] == 'H','pKaExp'], result.loc[result['Let'] == 'H','pKaComp'], linestyle = 'None', marker = 's', color = 'y', label = 'His', markeredgecolor = 'k', zorder = 5)
lys_p, = ax2.plot(result.loc[result['Let'] == 'K','pKaExp'], result.loc[result['Let'] == 'K','pKaComp'], linestyle = 'None', marker = 'D', color = 'c', label = 'Lys', markeredgecolor = 'k', zorder = 6)

ax2.grid(True, which = 'both')
ax2.plot([0,13],[0,13], linestyle = '-', color = 'black')
ax2.plot([0,13],[0,0], linestyle = '-', color = 'black')
ax2.plot([0,0],[0,13], linestyle = '-', color = 'black')
ax2.plot([1,13],[0,12], linestyle = '--', color = 'green')
ax2.plot([0,12],[1,13], linestyle = '--', color = 'green')

ax2.set_xlabel('pKa Exp', fontsize = 15)
ax2.set_ylabel('pKa Comp', fontsize = 15)

ax2.set_xticks(np.arange(0, 14, step=1))
ax2.set_yticks(np.arange(0, 14, step=1))
ax2.set_xlim([0,13])
ax2.set_ylim([0,13])

ax2.tick_params(axis = 'both', which = 'major', labelsize = 15)

legend_2 = ax2.legend(numpoints = 1, fontsize = 15, title = 'Tipo de Residuo', loc = 'upper left')
plt.setp(legend_2.get_title(), fontsize = 15)
#'''

'''
#######################################################################################################
#
# Plot Data - Error Comparison 
#
#######################################################################################################

fig, ax1 = plt.subplots()

# Error Comparison Plot

glu_err = np.abs(np.array(result.loc[result['Let'] == 'E','ShiftExp'] - result.loc[result['Let'] == 'E','ShiftComp']))
glu_null = np.abs(np.array(result.loc[result['Let'] == 'E','ShiftExp']))
asp_err = np.abs(np.array(result.loc[result['Let'] == 'D','ShiftExp'] - result.loc[result['Let'] == 'D','ShiftComp']))
asp_null = np.abs(np.array(result.loc[result['Let'] == 'D','ShiftExp']))
his_err = np.abs(np.array(result.loc[result['Let'] == 'H','ShiftExp'] - result.loc[result['Let'] == 'H','ShiftComp']))
his_null = np.abs(np.array(result.loc[result['Let'] == 'H','ShiftExp']))
lys_err = np.abs(np.array(result.loc[result['Let'] == 'K','ShiftExp'] - result.loc[result['Let'] == 'K','ShiftComp']))
lys_null = np.abs(np.array(result.loc[result['Let'] == 'K','ShiftExp']))

glu_s, = ax1.plot(glu_null,glu_err, linestyle = 'None', marker = 'o', color = 'b', label = 'Glu', markeredgecolor = 'k', zorder = 3)
asp_s, = ax1.plot(asp_null,asp_err, linestyle = 'None', marker = '^', color = 'r', label = 'Asp', markeredgecolor = 'k', zorder = 3)
his_s, = ax1.plot(his_null,his_err, linestyle = 'None', marker = 's', color = 'y', label = 'His', markeredgecolor = 'k', zorder = 3)
lys_s, = ax1.plot(lys_null,lys_err, linestyle = 'None', marker = 'D', color = 'c', label = 'Lys', markeredgecolor = 'k', zorder = 3)

ax1.grid(True, which = 'both')
ax1.set_yticks(np.arange(-5, 6, step=1))
ax1.set_xlim([-0.1,5])
ax1.set_ylim([-0.1,5])
ax1.plot([-10,400],[0,0], linestyle = '-', color = 'black')
ax1.plot([0,0],[-10,400], linestyle = '-', color = 'black')
ax1.plot([-10,400],[-10,400], linestyle = '-', color = 'black')
ax1.plot([-10,400],[1,1], linestyle = '--', color = 'green')
ax1.plot([1,1],[-10,400], linestyle = '--', color = 'green')
ax1.set_xlabel('Error del Modelo Nulo', fontsize = 15)
ax1.set_ylabel('Error del Ajuste', fontsize = 15)

ax1.tick_params(axis = 'both', which = 'major', labelsize = 15)

legend_1 = ax1.legend(numpoints = 1, fontsize = 15, title = 'Tipo de Residuo', loc = 'upper left')
plt.setp(legend_1.get_title(), fontsize = 15)

plt.show()
#'''
'''
#######################################################################################################
#
# Plot Data - Error Profile
#
#######################################################################################################

fig, ax1 = plt.subplots()

# Error Profile Plot

glu_err = np.abs(np.array(result.loc[result['Let'] == 'E','ShiftExp'] - result.loc[result['Let'] == 'E','ShiftComp']))
glu_null = np.abs(np.array(result.loc[result['Let'] == 'E','ShiftExp']))
glu_index = np.argsort(glu_err)
glu_err = glu_err[glu_index]
glu_null = glu_null[glu_index]

asp_err = np.abs(np.array(result.loc[result['Let'] == 'D','ShiftExp'] - result.loc[result['Let'] == 'D','ShiftComp']))
asp_null = np.abs(np.array(result.loc[result['Let'] == 'D','ShiftExp']))
asp_index = np.argsort(asp_err)
asp_err = asp_err[asp_index]
asp_null = asp_null[asp_index]

his_err = np.abs(np.array(result.loc[result['Let'] == 'H','ShiftExp'] - result.loc[result['Let'] == 'H','ShiftComp']))
his_null = np.abs(np.array(result.loc[result['Let'] == 'H','ShiftExp']))
his_index = np.argsort(his_err)
his_err = his_err[his_index]
his_null = his_null[his_index]

lys_err = np.abs(np.array(result.loc[result['Let'] == 'K','ShiftExp'] - result.loc[result['Let'] == 'K','ShiftComp']))
lys_null = np.abs(np.array(result.loc[result['Let'] == 'K','ShiftExp']))
lys_index = np.argsort(lys_err)
lys_err = lys_err[lys_index]
lys_null = lys_null[lys_index]

glu_s, = ax1.plot(glu_err, linestyle = 'None', marker = 'o', color = 'b', label = 'Glu', markeredgecolor = 'k', zorder = 3)
asp_s, = ax1.plot(asp_err, linestyle = 'None', marker = '^', color = 'r', label = 'Asp', markeredgecolor = 'k', zorder = 3)
his_s, = ax1.plot(his_err, linestyle = 'None', marker = 's', color = 'y', label = 'His', markeredgecolor = 'k', zorder = 3)
lys_s, = ax1.plot(lys_err, linestyle = 'None', marker = 'D', color = 'c', label = 'Lys', markeredgecolor = 'k', zorder = 3)

ax1.grid(True, which = 'both')
ax1.set_yticks(np.arange(-5, 6, step=1))
ax1.set_xlim([-5,len(glu_err) + 5])
ax1.set_ylim([-0.5,4])
ax1.plot([-10,400],[0,0], linestyle = '-', color = 'black')
ax1.plot([-10,400],[1,1], linestyle = '--', color = 'green')
ax1.plot([-10,400],[-1,-1], linestyle = '--', color = 'green')
ax1.set_ylabel('Error Absoluto', fontsize = 15)
ax1.set_xlabel('Indice Arbitrario', fontsize = 15)

ax1.tick_params(axis = 'both', which = 'major', labelsize = 15)

legend_1 = ax1.legend(numpoints = 1, fontsize = 15, title = 'Tipo de Residuo', loc = 'upper left')
plt.setp(legend_1.get_title(), fontsize = 15)

plt.show()

#'''
'''
#glu_s, = ax1.plot(glu_null, linestyle = 'None', marker = 'o', color = 'gray', label = 'Modelo Nulo', markeredgecolor = 'k', zorder = 3)

#asp_s, = ax1.plot(np.sort(np.abs(result.loc[result['Let'] == 'D','ShiftExp'] - result.loc[result['Let'] == 'D','ShiftComp'])), linestyle = 'None', marker = '^', color = 'r', label = 'Asp', markeredgecolor = 'k', zorder = 4)
#asp_s, = ax1.plot(np.sort(np.abs(result.loc[result['Let'] == 'D','pKaExp'] - result.loc[result['Let'] == 'D','pKaWat'])), linestyle = 'None', marker = '^', color = 'gray', label = 'Asp', markeredgecolor = 'k', zorder = 4)


#his_s, = ax1.plot(np.sort(result.loc[result['Let'] == 'H','ShiftExp'] - result.loc[result['Let'] == 'H','ShiftComp']), linestyle = 'None', marker = 's', color = 'y', label = 'His', markeredgecolor = 'k', zorder = 5)
#his_s, = ax1.plot(np.sort(result.loc[result['Let'] == 'H','pKaExp'] - result.loc[result['Let'] == 'H','pKaWat']), linestyle = 'None', marker = 's', color = 'gray', label = 'His', markeredgecolor = 'k', zorder = 5)

#lys_s, = ax1.plot(np.sort(result.loc[result['Let'] == 'K','ShiftExp'] - result.loc[result['Let'] == 'K','ShiftComp']), linestyle = 'None', marker = 'D', color = 'c', label = 'Lys', markeredgecolor = 'k', zorder = 6)
#lys_s, = ax1.plot(np.sort(result.loc[result['Let'] == 'K','pKaExp'] - result.loc[result['Let'] == 'K','pKaWat']), linestyle = 'None', marker = 'D', color = 'gray', label = 'Lys', markeredgecolor = 'k', zorder = 6)
'''

'''
# Text
null_model_text = ax1.text(2, -3.3, s = "Null RMSD: %.3f" % (null_model), fontsize = 15, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))
error_text = ax1.text(2, -4.5, s = "Fit RMSD: %.3f" % (rmsd), fontsize = 15, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))
'''
'''
#######################################################################################################
#
# Linear fit
#
#######################################################################################################

slope, intercept, r_value, p_value, std_err = stats.linregress(data['pKaExp']-data['pKaWat'],data['pKaComp']-data['pKaWat'])
xfit = np.arange(-5,6)
yfit = slope * xfit + intercept

fit, = plt.plot(xfit, yfit, linestyle = '-', color = 'red', label = 'Linear Fit')

fit_text = plt.text(-3, 3.8, s = "Slope: %.2f, Intercept: %.2f, R^2: %.2f" % (slope, intercept, r_value**2), fontsize = 15, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))
'''
'''
#######################################################################################################
#
# Histogram - Polar Neighbours
#
#######################################################################################################

# Source
# https://stackoverflow.com/questions/34933905/matplotlib-adding-subplots-to-a-subplot

# Bins

bns = np.linspace(0,8,17)

# Histograms

fig = plt.figure()
outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)

inner_all = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0], wspace=0.1, hspace=0.1)
ax_all = plt.Subplot(fig, inner_all[0])
all_s = ax_all.hist(result['Np'], rwidth = 0.85,
                    color = 'g', edgecolor = 'k', bins = bns,
                    label = 'Todos', zorder = 3, alpha = 0.9)
ax_all.set_xlabel('Vecinos Polares', fontsize = 15)
ax_all.set_ylabel('Cantidad de datos', fontsize = 15)
ax_all.tick_params(axis = 'both', which = 'major', labelsize = 15)
ax_all.set_xlim([-0.1,8])
ax_all.set_xticks(np.arange(0, 9, step = 1))
legend_all = ax_all.legend(numpoints = 1,
                           fontsize = 15,
                           title = 'Tipo de Residuo',
                           loc = 'upper right')
plt.setp(legend_all.get_title(), fontsize = 15)
fig.add_subplot(ax_all)

inner_types = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1], wspace=0.3, hspace=0.3)
ax_glu = plt.Subplot(fig, inner_types[0])
glu_s = ax_glu.hist(result.loc[result['Let'] == 'E','Np'], rwidth = 0.85,
                    color = 'b', edgecolor = 'k', bins = bns,
                    label = 'Glu', zorder = 3, alpha = 0.9)
ax_glu.tick_params(axis = 'both', which = 'major', labelsize = 10)
ax_glu.set_xlim([-0.1,8])
ax_glu.set_xticks(np.arange(0, 9, step = 1))
legend_glu = ax_glu.legend(numpoints = 1,
                           fontsize = 10,
                           loc = 'upper right')
plt.setp(legend_glu.get_title(), fontsize = 10)

ax_asp = plt.Subplot(fig, inner_types[1])
asp_s = ax_asp.hist(result.loc[result['Let'] == 'D','Np'], rwidth = 0.85,
                    color = 'r', edgecolor = 'k', bins = bns,
                    label = 'Asp', zorder = 4, alpha = 0.9)
ax_asp.tick_params(axis = 'both', which = 'major', labelsize = 10)
ax_asp.set_xlim([-0.1,8])
ax_asp.set_xticks(np.arange(0, 9, step = 1))
legend_asp = ax_asp.legend(numpoints = 1,
                           fontsize = 10,
                           loc = 'upper right')
plt.setp(legend_asp.get_title(), fontsize = 10)

ax_his = plt.Subplot(fig, inner_types[2])
his_s = ax_his.hist(result.loc[result['Let'] == 'H','Np'], rwidth = 0.85,
                    color = 'y', edgecolor = 'k', bins = bns,
                    label = 'His', zorder = 5, alpha = 0.9)
ax_his.tick_params(axis = 'both', which = 'major', labelsize = 10)
ax_his.set_xlim([-0.1,8])
ax_his.set_xticks(np.arange(0, 9, step = 1))
legend_his = ax_his.legend(numpoints = 1,
                           fontsize = 10,
                           loc = 'upper right')
plt.setp(legend_his.get_title(), fontsize = 10)

ax_lys = plt.Subplot(fig, inner_types[3])
lys_s = ax_lys.hist(result.loc[result['Let'] == 'K','Np'], rwidth = 0.85,
                    color = 'c', edgecolor = 'k', bins = bns,
                    label = 'Lys', zorder = 6, alpha = 0.9)
ax_lys.tick_params(axis = 'both', which = 'major', labelsize = 10)
ax_lys.set_xlim([-0.1,8])
ax_lys.set_xticks(np.arange(0, 9, step = 1))
legend_lys = ax_lys.legend(numpoints = 1,
                           fontsize = 10,
                           loc = 'upper right')
plt.setp(legend_lys.get_title(), fontsize = 10)

fig.add_subplot(ax_glu)
fig.add_subplot(ax_asp)
fig.add_subplot(ax_his)
fig.add_subplot(ax_lys)

fig.show()
#'''
'''
#######################################################################################################
#
# Histogram - Non Polar Neighbours
#
#######################################################################################################

# Bins

bns = np.linspace(0,20,40)

# Histograms

fig = plt.figure()
outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)

inner_all = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0], wspace=0.1, hspace=0.1)
ax_all = plt.Subplot(fig, inner_all[0])
all_s = ax_all.hist(result['Nnp'], rwidth = 0.85,
                    color = 'g', edgecolor = 'k', bins = bns,
                    label = 'Todos', zorder = 3, alpha = 0.9)
ax_all.set_xlabel('Vecinos No Polares', fontsize = 15)
ax_all.set_ylabel('Cantidad de datos', fontsize = 15)
ax_all.tick_params(axis = 'both', which = 'major', labelsize = 15)
ax_all.set_xlim([-0.1,20])
ax_all.set_xticks(np.arange(0, 22, step = 2))
legend_all = ax_all.legend(numpoints = 1,
                           fontsize = 15,
                           title = 'Tipo de Residuo',
                           loc = 'upper right')
plt.setp(legend_all.get_title(), fontsize = 15)
fig.add_subplot(ax_all)

inner_types = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1], wspace=0.3, hspace=0.3)
ax_glu = plt.Subplot(fig, inner_types[0])
glu_s = ax_glu.hist(result.loc[result['Let'] == 'E','Nnp'], rwidth = 0.85,
                    color = 'b', edgecolor = 'k', bins = bns,
                    label = 'Glu', zorder = 3, alpha = 0.9)
ax_glu.tick_params(axis = 'both', which = 'major', labelsize = 9)
ax_glu.set_xlim([-0.1,20])
ax_glu.set_xticks(np.arange(0, 22, step = 2))
legend_glu = ax_glu.legend(numpoints = 1,
                           fontsize = 10,
                           loc = 'upper right')
plt.setp(legend_glu.get_title(), fontsize = 10)

ax_asp = plt.Subplot(fig, inner_types[1])
asp_s = ax_asp.hist(result.loc[result['Let'] == 'D','Nnp'], rwidth = 0.85,
                    color = 'r', edgecolor = 'k', bins = bns,
                    label = 'Asp', zorder = 4, alpha = 0.9)
ax_asp.tick_params(axis = 'both', which = 'major', labelsize = 9)
ax_asp.set_xlim([-0.1,20])
ax_asp.set_xticks(np.arange(0, 22, step = 2))
legend_asp = ax_asp.legend(numpoints = 1,
                           fontsize = 10,
                           loc = 'upper right')
plt.setp(legend_asp.get_title(), fontsize = 10)

ax_his = plt.Subplot(fig, inner_types[2])
his_s = ax_his.hist(result.loc[result['Let'] == 'H','Nnp'], rwidth = 0.85,
                    color = 'y', edgecolor = 'k', bins = bns,
                    label = 'His', zorder = 5, alpha = 0.9)
ax_his.tick_params(axis = 'both', which = 'major', labelsize = 9)
ax_his.set_xlim([-0.1,20])
ax_his.set_xticks(np.arange(0, 22, step = 2))
legend_his = ax_his.legend(numpoints = 1,
                           fontsize = 10,
                           loc = 'upper right')
plt.setp(legend_his.get_title(), fontsize = 10)

ax_lys = plt.Subplot(fig, inner_types[3])
lys_s = ax_lys.hist(result.loc[result['Let'] == 'K','Nnp'], rwidth = 0.85,
                    color = 'c', edgecolor = 'k', bins = bns,
                    label = 'Lys', zorder = 6, alpha = 0.9)
ax_lys.tick_params(axis = 'both', which = 'major', labelsize = 9)
ax_lys.set_xlim([-0.1,20])
ax_lys.set_xticks(np.arange(0, 22, step = 2))
legend_lys = ax_lys.legend(numpoints = 1,
                           fontsize = 10,
                           loc = 'upper right')
plt.setp(legend_lys.get_title(), fontsize = 10)

fig.add_subplot(ax_glu)
fig.add_subplot(ax_asp)
fig.add_subplot(ax_his)
fig.add_subplot(ax_lys)

fig.show()

#'''
'''
#######################################################################################################
#
# Histogram - Ionizable Neighbours
#
#######################################################################################################

# Bins

bns = np.linspace(0,6,12)

# Histograms

fig = plt.figure()
outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)

inner_all = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0], wspace=0.1, hspace=0.1)
ax_all = plt.Subplot(fig, inner_all[0])
all_s = ax_all.hist(result['Ni'], rwidth = 0.85,
                    color = 'g', edgecolor = 'k', bins = bns,
                    label = 'Todos', zorder = 3, alpha = 0.9)
ax_all.set_xlabel('Vecinos Ionizables', fontsize = 15)
ax_all.set_ylabel('Cantidad de datos', fontsize = 15)
ax_all.tick_params(axis = 'both', which = 'major', labelsize = 15)
ax_all.set_xlim([-0.1,6])
ax_all.set_xticks(np.arange(0, 7, step = 1))
legend_all = ax_all.legend(numpoints = 1,
                           fontsize = 15,
                           title = 'Tipo de Residuo',
                           loc = 'upper right')
plt.setp(legend_all.get_title(), fontsize = 15)
fig.add_subplot(ax_all)

inner_types = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1], wspace=0.3, hspace=0.3)
ax_glu = plt.Subplot(fig, inner_types[0])
glu_s = ax_glu.hist(result.loc[result['Let'] == 'E','Ni'], rwidth = 0.85,
                    color = 'b', edgecolor = 'k', bins = bns,
                    label = 'Glu', zorder = 3, alpha = 0.9)
ax_glu.tick_params(axis = 'both', which = 'major', labelsize = 10)
ax_glu.set_xlim([-0.1,6])
ax_glu.set_xticks(np.arange(0, 7, step = 1))
legend_glu = ax_glu.legend(numpoints = 1,
                           fontsize = 10,
                           loc = 'upper right')
plt.setp(legend_glu.get_title(), fontsize = 10)

ax_asp = plt.Subplot(fig, inner_types[1])
asp_s = ax_asp.hist(result.loc[result['Let'] == 'D','Ni'], rwidth = 0.85,
                    color = 'r', edgecolor = 'k', bins = bns,
                    label = 'Asp', zorder = 4, alpha = 0.9)
ax_asp.tick_params(axis = 'both', which = 'major', labelsize = 10)
ax_asp.set_xlim([-0.1,6])
ax_asp.set_xticks(np.arange(0, 7, step = 1))
legend_asp = ax_asp.legend(numpoints = 1,
                           fontsize = 10,
                           loc = 'upper right')
plt.setp(legend_asp.get_title(), fontsize = 10)

ax_his = plt.Subplot(fig, inner_types[2])
his_s = ax_his.hist(result.loc[result['Let'] == 'H','Ni'], rwidth = 0.85,
                    color = 'y', edgecolor = 'k', bins = bns,
                    label = 'His', zorder = 5, alpha = 0.9)
ax_his.tick_params(axis = 'both', which = 'major', labelsize = 10)
ax_his.set_xlim([-0.1,6])
ax_his.set_xticks(np.arange(0, 7, step = 1))
legend_his = ax_his.legend(numpoints = 1,
                           fontsize = 10,
                           loc = 'upper right')
plt.setp(legend_his.get_title(), fontsize = 10)

ax_lys = plt.Subplot(fig, inner_types[3])
lys_s = ax_lys.hist(result.loc[result['Let'] == 'K','Ni'], rwidth = 0.85,
                    color = 'c', edgecolor = 'k', bins = bns,
                    label = 'Lys', zorder = 6, alpha = 0.9)
ax_lys.tick_params(axis = 'both', which = 'major', labelsize = 10)
ax_lys.set_xlim([-0.1,6])
ax_lys.set_xticks(np.arange(0, 7, step = 1))
legend_lys = ax_lys.legend(numpoints = 1,
                           fontsize = 10,
                           loc = 'upper right')
plt.setp(legend_lys.get_title(), fontsize = 10)

fig.add_subplot(ax_glu)
fig.add_subplot(ax_asp)
fig.add_subplot(ax_his)
fig.add_subplot(ax_lys)

fig.show()
#'''
'''
#######################################################################################################
#
# Histogram - Shift Exp
#
#######################################################################################################

# Bins

bns = np.linspace(-5,5,61)

# Histograms

fig = plt.figure()
outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)

inner_all = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0], wspace=0.1, hspace=0.1)
ax_all = plt.Subplot(fig, inner_all[0])
all_s = ax_all.hist(result['ShiftExp'], rwidth = 0.85,
                    color = 'g', edgecolor = 'k', bins = bns,
                    label = 'Todos', zorder = 3, alpha = 0.9)
ax_all.set_xlabel('Corrimiento de pKa', fontsize = 15)
ax_all.set_ylabel('Cantidad de datos', fontsize = 15)
ax_all.tick_params(axis = 'both', which = 'major', labelsize = 15)
ax_all.set_xlim([-5.1,5.1])
ax_all.set_xticks(np.arange(-5, 6, step = 1))
legend_all = ax_all.legend(numpoints = 1,
                           fontsize = 15,
                           title = 'Tipo de Residuo',
                           loc = 'upper right')
plt.setp(legend_all.get_title(), fontsize = 15)
fig.add_subplot(ax_all)

inner_types = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1], wspace=0.3, hspace=0.3)
ax_glu = plt.Subplot(fig, inner_types[0])
glu_s = ax_glu.hist(result.loc[result['Let'] == 'E','ShiftExp'], rwidth = 0.85,
                    color = 'b', edgecolor = 'k', bins = bns,
                    label = 'Glu', zorder = 3, alpha = 0.9)
ax_glu.tick_params(axis = 'both', which = 'major', labelsize = 10)
ax_glu.set_xlim([-5.1,5.1])
ax_glu.set_xticks(np.arange(-5, 6, step = 1))
legend_glu = ax_glu.legend(numpoints = 1,
                           fontsize = 10,
                           loc = 'upper right')
plt.setp(legend_glu.get_title(), fontsize = 10)

ax_asp = plt.Subplot(fig, inner_types[1])
asp_s = ax_asp.hist(result.loc[result['Let'] == 'D','ShiftExp'], rwidth = 0.85,
                    color = 'r', edgecolor = 'k', bins = bns,
                    label = 'Asp', zorder = 4, alpha = 0.9)
ax_asp.tick_params(axis = 'both', which = 'major', labelsize = 10)
ax_asp.set_xlim([-5.1,5.1])
ax_asp.set_xticks(np.arange(-5, 6, step = 1))
legend_asp = ax_asp.legend(numpoints = 1,
                           fontsize = 10,
                           loc = 'upper right')
plt.setp(legend_asp.get_title(), fontsize = 10)

ax_his = plt.Subplot(fig, inner_types[2])
his_s = ax_his.hist(result.loc[result['Let'] == 'H','ShiftExp'], rwidth = 0.85,
                    color = 'y', edgecolor = 'k', bins = bns,
                    label = 'His', zorder = 5, alpha = 0.9)
ax_his.tick_params(axis = 'both', which = 'major', labelsize = 10)
ax_his.set_xlim([-5.1,5.1])
ax_his.set_xticks(np.arange(-5, 6, step = 1))
legend_his = ax_his.legend(numpoints = 1,
                           fontsize = 10,
                           loc = 'upper right')
plt.setp(legend_his.get_title(), fontsize = 10)

ax_lys = plt.Subplot(fig, inner_types[3])
lys_s = ax_lys.hist(result.loc[result['Let'] == 'K','ShiftExp'], rwidth = 0.85,
                    color = 'c', edgecolor = 'k', bins = bns,
                    label = 'Lys', zorder = 6, alpha = 0.9)
ax_lys.tick_params(axis = 'both', which = 'major', labelsize = 10)
ax_lys.set_xlim([-5.1,5.1])
ax_lys.set_xticks(np.arange(-5, 6, step = 1))
legend_lys = ax_lys.legend(numpoints = 1,
                           fontsize = 10,
                           loc = 'upper right')
plt.setp(legend_lys.get_title(), fontsize = 10)

fig.add_subplot(ax_glu)
fig.add_subplot(ax_asp)
fig.add_subplot(ax_his)
fig.add_subplot(ax_lys)

fig.show()
#'''

plt.show()
