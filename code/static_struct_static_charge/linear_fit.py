import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

######################################################
# Constant pH model with linear dependence
# This function is used to fit the linear parameters
######################################################
def linear_ph_model(params, e, up, unp, pkawat, pkaexp):
    pkacomp = pkawat + params[0] * e + params[1] * up + params[2] * unp

    residual = pkacomp - pkaexp

    return residual

######################################################
# Constant pH model with linear dependence
# This function is used to plot
######################################################
def linear_ph_model_plot(params, e, up, unp, pkawat, pkaexp):
    pkacomp = pkawat + params[0] * e + params[1] * up + params[2] * unp

    return pkacomp

####################
# Group Definition
####################
s_min = -1
s_max = 1

################
# Data and Fit
################

# Read Datapoints from two configurations, steep and soft
data_steep = pd.read_csv('result_matrix_steep.txt', sep = '\t', index_col = 0)
data_soft = pd.read_csv('result_matrix_soft.txt', sep = '\t', index_col = 0)

# All Datapoints included
# Steep
elec_all_steep = np.array(data_steep['Elec'])
up_all_steep = np.array(data_steep['Up'])
unp_all_steep = np.array(data_steep['Unp'])
pka_wat_all_steep = np.array(data_steep['pKaWat'])
pka_exp_all_steep = np.array(data_steep['pKaExp'])



# Soft
elec_all_soft = np.array(data_soft['Elec'])
up_all_soft = np.array(data_soft['Up'])
unp_all_soft = np.array(data_soft['Unp'])
pka_wat_all_soft = np.array(data_soft['pKaWat'])
pka_exp_all_soft = np.array(data_soft['pKaExp'])

# Initial guess and Bounds on Parameters
initial_guess = [3, 3, 3]
bnds = [(0,0,0),(15,15,15)]

# Fitting ALL Datapoints
result_all_soft = optimize.least_squares(linear_ph_model,
                                         x0 = initial_guess,
                                         args = (elec_all_soft,
                                                 up_all_soft,
                                                 unp_all_soft,
                                                 pka_wat_all_soft,
                                                 pka_exp_all_soft),
                                         bounds = bnds,
                                         loss = 'soft_l1', f_scale = 0.5)

result_all_steep = optimize.least_squares(linear_ph_model,
                                          x0 = initial_guess,
                                          args = (elec_all_steep,
                                                  up_all_steep,
                                                  unp_all_steep,
                                                  pka_wat_all_steep,
                                                  pka_exp_all_steep),
                                          bounds = bnds,
                                          loss = 'soft_l1', f_scale = 0.5)


# Only IN Datapoints, range --->  s_min < ShiftExp < s_max

# Soft
elec_in_soft    = np.array(data_soft.loc[(data_soft['ShiftExp'] < s_max) & (data_soft['ShiftExp'] > s_min),'Elec'])
up_in_soft      = np.array(data_soft.loc[(data_soft['ShiftExp'] < s_max) & (data_soft['ShiftExp'] > s_min),'Up'])
unp_in_soft     = np.array(data_soft.loc[(data_soft['ShiftExp'] < s_max) & (data_soft['ShiftExp'] > s_min),'Unp'])
pka_wat_in_soft = np.array(data_soft.loc[(data_soft['ShiftExp'] < s_max) & (data_soft['ShiftExp'] > s_min),'pKaWat'])
pka_exp_in_soft = np.array(data_soft.loc[(data_soft['ShiftExp'] < s_max) & (data_soft['ShiftExp'] > s_min),'pKaExp'])

# Steep
elec_in_steep    = np.array(data_steep.loc[(data_steep['ShiftExp'] < s_max) & (data_steep['ShiftExp'] > s_min),'Elec'])
up_in_steep      = np.array(data_steep.loc[(data_steep['ShiftExp'] < s_max) & (data_steep['ShiftExp'] > s_min),'Up'])
unp_in_steep     = np.array(data_steep.loc[(data_steep['ShiftExp'] < s_max) & (data_steep['ShiftExp'] > s_min),'Unp'])
pka_wat_in_steep = np.array(data_steep.loc[(data_steep['ShiftExp'] < s_max) & (data_steep['ShiftExp'] > s_min),'pKaWat'])
pka_exp_in_steep = np.array(data_steep.loc[(data_steep['ShiftExp'] < s_max) & (data_steep['ShiftExp'] > s_min),'pKaExp'])

# Fitting
result_in_soft = optimize.least_squares(linear_ph_model,
                                        x0 = initial_guess,
                                        args = (elec_in_soft,
                                                up_in_soft,
                                                unp_in_soft,
                                                pka_wat_in_soft,
                                                pka_exp_in_soft),
                                        bounds = bnds,
                                        loss = 'soft_l1', f_scale = 0.5)

result_in_steep = optimize.least_squares(linear_ph_model,
                                         x0 = initial_guess,
                                         args = (elec_in_steep,
                                                 up_in_steep,
                                                 unp_in_steep,
                                                 pka_wat_in_steep,
                                                 pka_exp_in_steep),
                                         bounds = bnds,
                                         loss = 'soft_l1', f_scale = 0.5)


# Only OUT Datapoints, range --->  ShiftExp < s_min and ShiftExp > s_max

# Soft
elec_out_soft    = np.array(data_soft.loc[(data_soft['ShiftExp'] >= s_max) | (data_soft['ShiftExp'] <= s_min),'Elec'])
up_out_soft      = np.array(data_soft.loc[(data_soft['ShiftExp'] >= s_max) | (data_soft['ShiftExp'] <= s_min),'Up'])
unp_out_soft     = np.array(data_soft.loc[(data_soft['ShiftExp'] >= s_max) | (data_soft['ShiftExp'] <= s_min),'Unp'])
pka_wat_out_soft = np.array(data_soft.loc[(data_soft['ShiftExp'] >= s_max) | (data_soft['ShiftExp'] <= s_min),'pKaWat'])
pka_exp_out_soft = np.array(data_soft.loc[(data_soft['ShiftExp'] >= s_max) | (data_soft['ShiftExp'] <= s_min),'pKaExp'])

# Steep
elec_out_steep    = np.array(data_steep.loc[(data_steep['ShiftExp'] >= s_max) | (data_steep['ShiftExp'] <= s_min),'Elec'])
up_out_steep      = np.array(data_steep.loc[(data_steep['ShiftExp'] >= s_max) | (data_steep['ShiftExp'] <= s_min),'Up'])
unp_out_steep     = np.array(data_steep.loc[(data_steep['ShiftExp'] >= s_max) | (data_steep['ShiftExp'] <= s_min),'Unp'])
pka_wat_out_steep = np.array(data_steep.loc[(data_steep['ShiftExp'] >= s_max) | (data_steep['ShiftExp'] <= s_min),'pKaWat'])
pka_exp_out_steep = np.array(data_steep.loc[(data_steep['ShiftExp'] >= s_max) | (data_steep['ShiftExp'] <= s_min),'pKaExp'])

# Fitting
result_out_soft = optimize.least_squares(linear_ph_model,
                                         x0 = initial_guess,
                                         args = (elec_out_soft,
                                                 up_out_soft,
                                                 unp_out_soft,
                                                 pka_wat_out_soft,
                                                 pka_exp_out_soft),
                                         bounds = bnds,
                                         loss = 'linear', f_scale = 1.0)

result_out_steep = optimize.least_squares(linear_ph_model,
                                          x0 = initial_guess,
                                          args = (elec_out_steep,
                                                  up_out_steep,
                                                  unp_out_steep,
                                                  pka_wat_out_steep,
                                                  pka_exp_out_steep),
                                          bounds = bnds,
                                          loss = 'linear', f_scale = 1.0)

print(len(pka_exp_all_soft))
print(len(pka_exp_in_soft))
print(len(pka_exp_out_soft))
#################
# Print Result
#################

rmsd_all_null = np.sqrt(np.sum(np.square(pka_exp_all_soft - pka_wat_all_soft)/len(pka_exp_all_soft)))
rmsd_in_null  = np.sqrt(np.sum(np.square(pka_exp_in_soft - pka_wat_in_soft)/len(pka_exp_in_soft)))
rmsd_out_null = np.sqrt(np.sum(np.square(pka_exp_out_soft - pka_wat_out_soft)/len(pka_exp_out_soft)))

# Rescaling
k_b = 0.0019872041 # kcal/(mol*K)
T = 300 # K

# All Soft
resid_all_soft = linear_ph_model(result_all_soft.x,
                                 elec_all_soft,
                                 up_all_soft,
                                 unp_all_soft,
                                 pka_wat_all_soft,
                                 pka_exp_all_soft)
rmsd_all_soft = np.sqrt(np.sum(np.square(resid_all_soft)/len(resid_all_soft)))
print("Result for ALL Datapoints and Soft Dataset")
print("Kelec: %.3f" % (result_all_soft.x[0] * k_b * T * np.log(10)))
print("Bp: %.3f" % (result_all_soft.x[1] * k_b * T * np.log(10)))
print("Bnp: %.3f" % (result_all_soft.x[2] * k_b * T * np.log(10)))
print("RMSD Fit: %.3f" % (rmsd_all_soft))
print("RMSD Null: %.3f" % (rmsd_all_null))

# All Steep
resid_all_steep = linear_ph_model(result_all_steep.x,
                                  elec_all_steep,
                                  up_all_steep,
                                  unp_all_steep,
                                  pka_wat_all_steep,
                                  pka_exp_all_steep)
rmsd_all_steep = np.sqrt(np.sum(np.square(resid_all_steep)/len(resid_all_steep)))
print("Result for ALL Datapoints and Steep Dataset")
print("Kelec: %.3f" % (result_all_steep.x[0] * k_b * T * np.log(10)))
print("Bp: %.3f" % (result_all_steep.x[1] * k_b * T * np.log(10)))
print("Bnp: %.3f" % (result_all_steep.x[2] * k_b * T * np.log(10)))
print("RMSD: %.3f" % (rmsd_all_steep))
print("RMSD Null: %.3f" % (rmsd_all_null))
      
# In Soft
resid_in_soft = linear_ph_model(result_in_soft.x,
                                elec_in_soft,
                                up_in_soft,
                                unp_in_soft,
                                pka_wat_in_soft,
                                pka_exp_in_soft)
rmsd_in_soft = np.sqrt(np.sum(np.square(resid_in_soft)/len(resid_in_soft)))                        
print("Result for IN Datapoints and Soft Dataset")
print("Kelec: %.3f" % (result_in_soft.x[0] * k_b * T * np.log(10)))
print("Bp: %.3f" % (result_in_soft.x[1] * k_b * T * np.log(10)))
print("Bnp: %.3f" % (result_in_soft.x[2] * k_b * T * np.log(10)))
print("RMSD Fit: %.3f" % (rmsd_in_soft))
print("RMSD Null: %.3f" % (rmsd_in_null))
      
# In Steep
resid_in_steep = linear_ph_model(result_in_steep.x,
                                 elec_in_steep,
                                 up_in_steep,
                                 unp_in_steep,
                                 pka_wat_in_steep,
                                 pka_exp_in_steep)
rmsd_in_steep = np.sqrt(np.sum(np.square(resid_in_steep)/len(resid_in_steep)))
print("Result for IN Datapoints and Steep Dataset")
print("Kelec: %.3f" % (result_in_steep.x[0] * k_b * T * np.log(10)))
print("Bp: %.3f" % (result_in_steep.x[1] * k_b * T * np.log(10)))
print("Bnp: %.3f" % (result_in_steep.x[2] * k_b * T * np.log(10)))
print("RMSD Fit: %.3f" % (rmsd_in_steep))
print("RMSD Null: %.3f" % (rmsd_in_null))
      
# Out Soft
resid_out_soft = linear_ph_model(result_out_soft.x,
                                 elec_out_soft,
                                 up_out_soft,
                                 unp_out_soft,
                                 pka_wat_out_soft,
                                 pka_exp_out_soft)
rmsd_out_soft = np.sqrt(np.sum(np.square(resid_out_soft)/len(resid_out_soft))) 
print("Result for OUT Datapoints and Soft Dataset")
print("Kelec: %.3f" % (result_out_soft.x[0] * k_b * T * np.log(10)))
print("Bp: %.3f" % (result_out_soft.x[1] * k_b * T * np.log(10)))
print("Bnp: %.3f" % (result_out_soft.x[2] * k_b * T * np.log(10)))
print("RMSD Fit: %.3f" % (rmsd_out_soft))
print("RMSD Null: %.3f" % (rmsd_out_null))
      
# Out Steep
resid_out_steep = linear_ph_model(result_out_steep.x,
                                 elec_out_steep,
                                 up_out_steep,
                                 unp_out_steep,
                                 pka_wat_out_steep,
                                 pka_exp_out_steep)
rmsd_out_steep = np.sqrt(np.sum(np.square(resid_out_steep)/len(resid_out_steep)))
print("Result for OUT Datapoints and Steep Dataset")
print("Kelec: %.3f" % (result_out_steep.x[0] * k_b * T * np.log(10)))
print("Bp: %.3f" % (result_out_steep.x[1] * k_b * T * np.log(10)))
print("Bnp: %.3f" % (result_out_steep.x[2] * k_b * T * np.log(10)))
print("RMSD Fit: %.3f" % (rmsd_out_steep))
print("RMSD Null: %.3f" % (rmsd_out_null))
      
################
# Plot Result
################

contrast = 0.75
mksize = 4
mkedgewidth = 0.75

##############################
# Neighbour Penalty Function
##############################
def neigh_penalty(n_list,ncut,alpha):
    y = np.zeros(len(n_list))
    for idx, n in enumerate(n_list):
        if n < ncut:
            val = np.exp(- alpha * (n - ncut)**2)
        else:
            val = 1.0
        y[idx] = val
    return y

f, ax1 = plt.subplots(1, 1)

# Penalty Function
ncut_p = 4.5
ncut_np = 10
alpha_soft = 0.02
alpha_steep = 5

neigh_x = np.linspace(0,15,100)
y_soft_polar = neigh_penalty(neigh_x,ncut_p,alpha_soft)
y_soft_non_polar = neigh_penalty(neigh_x,ncut_np,alpha_soft)

y_steep_polar = neigh_penalty(neigh_x,ncut_p,alpha_steep)
y_steep_non_polar = neigh_penalty(neigh_x,ncut_np,alpha_steep)

ax1.set_title("Penalidad por Polaridad")
ax1.plot(neigh_x, y_soft_polar, color = 'green', linestyle = '--', linewidth = 3, label = 'Polar Suave')
ax1.plot(neigh_x, y_soft_non_polar, color = 'orange',linestyle = '--', linewidth = 3, label = 'No Polar Suave')

ax1.plot(neigh_x, y_steep_polar, color = 'green', linewidth = 3, label = 'Polar Empinado')
ax1.plot(neigh_x, y_steep_non_polar, color = 'orange', linewidth = 3, label = 'No Polar Empinado')

ax1.set_xlabel("Vecinos", fontsize = 15)
ax1.set_ylabel("Penalidad", fontsize = 15)
ax1.set_xlim([0,15])
ax1.set_ylim([-0.1,1.1])
ax1.set_xticks(np.arange(0,16,2))
ax1.set_yticks(np.arange(0,1.2,0.2))
ax1.tick_params(axis = 'both', which = 'major', labelsize = 15)

ax1.grid(True, which = 'both')
ax1.legend(loc = 'center right', fontsize = 12)

# Fit Plots
f2, ((ax2, ax3, ax4)) = plt.subplots(1, 3, sharey = True)
f2.subplots_adjust(wspace = 0.1)
f2.suptitle('Ajuste lineal del Modelo de pH Constante')
# ALL
ax2.set_title("Todos")
ax2.plot(pka_exp_all_soft - pka_wat_all_soft,
         linear_ph_model_plot(result_all_soft.x,
                              elec_all_soft,
                              up_all_soft,
                              unp_all_soft,
                              pka_wat_all_soft,
                              pka_exp_all_soft) - pka_wat_all_soft,
         marker = 'o', color = 'b', label = 'Suave', linestyle = 'None',
         markeredgecolor = 'k', alpha = contrast,
         markersize = mksize, markeredgewidth = mkedgewidth)

ax2.plot(pka_exp_all_steep - pka_wat_all_steep,
         linear_ph_model_plot(result_all_steep.x,
                              elec_all_steep,
                              up_all_steep,
                              unp_all_steep,
                              pka_wat_all_steep,
                              pka_exp_all_steep) - pka_wat_all_steep,
         marker = 'D', color = 'r', label = 'Empinado', linestyle = 'None',
         markeredgecolor = 'k', alpha = contrast,
         markersize = mksize, markeredgewidth = mkedgewidth)

ax2.set_xlabel("Corrimiento Experimental")
ax2.set_ylabel("Corrimiento Calculado")
ax2.set_xlim([-5.2,5.2])
ax2.set_ylim([-5.2,5.2])
ax2.set_xticks(np.arange(-5,6,1))
ax2.set_yticks(np.arange(-5,6,1))
ax2.plot([-6,6],[-6,6], linestyle = '-', color = 'black')
ax2.plot([-6,6],[0,0], linestyle = '-', color = 'black')
ax2.plot([0,0],[-6,6], linestyle = '-', color = 'black')
ax2.plot([-5,6],[-6,5], linestyle = '--', color = 'green')
ax2.plot([-6,5],[-5,6], linestyle = '--', color = 'green')
ax2.grid(True, which = 'both')
ax2.yaxis.set_ticklabels(np.arange(-5,6,1))
ax2.legend(numpoints = 1, loc = 'upper left')
ax2.set_aspect('equal')

# IN
ax3.set_title("Dentro")
'''
ax3.plot(pka_exp_in_soft - pka_wat_in_soft,
         linear_ph_model_plot(result_in_soft.x,
                              elec_in_soft,
                              up_in_soft,
                              unp_in_soft,
                              pka_wat_in_soft,
                              pka_exp_in_soft) - pka_wat_in_soft,
         marker = 'o', color = 'b', label = 'Suave', linestyle = 'None',
         markeredgecolor = 'k', alpha = contrast,
         markersize = mksize, markeredgewidth = mkedgewidth)
'''
ax3.plot(pka_exp_in_steep - pka_wat_in_steep,
         linear_ph_model_plot(result_in_steep.x,
                              elec_in_steep,
                              up_in_steep,
                              unp_in_steep,
                              pka_wat_in_steep,
                              pka_exp_in_steep) - pka_wat_in_steep,
         marker = 'D', color = 'r', label = 'Empinado', linestyle = 'None',
         markeredgecolor = 'k', alpha = contrast,
         markersize = mksize, markeredgewidth = mkedgewidth)

ax3.set_xlabel("Corrimiento Experimental")
ax3.set_xlim([-5.2,5.2])
ax3.set_ylim([-5.2,5.2])
ax3.set_xticks(np.arange(-5,6,1))
ax3.set_yticks(np.arange(-5,6,1))
ax3.plot([-6,6],[-6,6], linestyle = '-', color = 'black')
ax3.plot([-6,6],[0,0], linestyle = '-', color = 'black')
ax3.plot([0,0],[-6,6], linestyle = '-', color = 'black')
ax3.plot([-5,6],[-6,5], linestyle = '--', color = 'green')
ax3.plot([-6,5],[-5,6], linestyle = '--', color = 'green')
ax3.grid(True, which = 'both')
#ax3.yaxis.set_ticklabels([])
ax3.legend(numpoints = 1, loc = 'upper left')

ax3.set_aspect('equal')

# OUT
ax4.set_title("Fuera")
ax4.plot(pka_exp_out_soft - pka_wat_out_soft,
         linear_ph_model_plot(result_out_soft.x,
                              elec_out_soft,
                              up_out_soft,
                              unp_out_soft,
                              pka_wat_out_soft,
                              pka_exp_out_soft) - pka_wat_out_soft,
         marker = 'o', color = 'b', label = 'Suave', linestyle = 'None',
         markeredgecolor = 'k', alpha = contrast,
         markersize = mksize, markeredgewidth = mkedgewidth)

ax4.plot(pka_exp_out_steep - pka_wat_out_steep,
         linear_ph_model_plot(result_out_steep.x,
                              elec_out_steep,
                              up_out_steep,
                              unp_out_steep,
                              pka_wat_out_steep,
                              pka_exp_out_steep) - pka_wat_out_steep,
         marker = 'D', color = 'r', label = 'Empinado', linestyle = 'None',
         markeredgecolor = 'k', alpha = contrast,
         markersize = mksize, markeredgewidth = mkedgewidth)

ax4.set_xlabel("Corrimiento Experimental")
ax4.set_xlim([-5.2,5.2])
ax4.set_ylim([-5.2,5.2])
ax4.set_xticks(np.arange(-5,6,1))
ax4.set_yticks(np.arange(-5,6,1))
ax4.plot([-6,6],[-6,6], linestyle = '-', color = 'black')
ax4.plot([-6,6],[0,0], linestyle = '-', color = 'black')
ax4.plot([0,0],[-6,6], linestyle = '-', color = 'black')
ax4.plot([-5,6],[-6,5], linestyle = '--', color = 'green')
ax4.plot([-6,5],[-5,6], linestyle = '--', color = 'green')
ax4.grid(True, which = 'both')
#ax4.yaxis.set_ticklabels([])
ax4.legend(numpoints = 1, loc = 'upper left')

ax4.set_aspect('equal')

plt.tight_layout()

plt.show()
