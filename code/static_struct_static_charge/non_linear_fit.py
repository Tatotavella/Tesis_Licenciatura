import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from model_to_fit import constant_ph_model

######################################################
# Constant pH model with non linear dependence
# This function is used to fit the linear parameters
######################################################
def non_linear_ph_model(params, data):
    
    Kelec = params[0]                                    # All
    Ap    = [params[1], params[2], params[3], params[4]] # [Asp, Glu, Lys-Arg, His]
    Anp   = [params[5], params[6], params[7], params[8]] # [Asp, Glu, Lys-Arg, His]
    NeighMax = [params[9], params[10]]                   # [Polar, Non-Polar]
    slope    = [params[11], params[12]]                  # [Polar, Non-Polar]

    pkacomp = constant_ph_model(data, Kelec, Ap, Anp, NeighMax, slope)
    
    residual = pkacomp - data['pKaExp']

    return residual


####################
# Group Definition
####################
s_min = -2
s_max = 1

################
# Data and Fit
################

# Read Datapoints from two configurations, steep and soft
data_all_steep = pd.read_csv('result_matrix_steep.txt', sep = '\t', index_col = 0)
data_all_soft = pd.read_csv('result_matrix_soft.txt', sep = '\t', index_col = 0)

# Null RMSD
pka_wat_all_soft = np.array(data_all_soft['pKaWat'])
pka_exp_all_soft = np.array(data_all_soft['pKaExp'])
pka_wat_in_soft  = np.array(data_all_soft.loc[(data_all_soft['ShiftExp'] < s_max) & (data_all_soft['ShiftExp'] > s_min),'pKaWat'])
pka_exp_in_soft  = np.array(data_all_soft.loc[(data_all_soft['ShiftExp'] < s_max) & (data_all_soft['ShiftExp'] > s_min),'pKaExp'])
pka_wat_out_soft = np.array(data_all_soft.loc[(data_all_soft['ShiftExp'] > s_max) | (data_all_soft['ShiftExp'] < s_min),'pKaWat'])
pka_exp_out_soft = np.array(data_all_soft.loc[(data_all_soft['ShiftExp'] > s_max) | (data_all_soft['ShiftExp'] < s_min),'pKaExp'])

rmsd_all_null = np.sqrt(np.sum(np.square(pka_exp_all_soft - pka_wat_all_soft)/len(pka_exp_all_soft)))
rmsd_in_null  = np.sqrt(np.sum(np.square(pka_exp_in_soft - pka_wat_in_soft)/len(pka_exp_in_soft)))
rmsd_out_null = np.sqrt(np.sum(np.square(pka_exp_out_soft - pka_wat_out_soft)/len(pka_exp_out_soft)))

# Rescaling
k_b = 0.0019872041 # kcal/(mol*K)
T   = 300          # K

# Initial guess and Bounds on Parameters
initial_guess_all_soft  = [3.539,
                           0.450, 0.450, 0.450, 0.450,
                           4.425, 4.425, 4.425, 4.425,
                           3.226, 12.752,
                           0.067, 10]

initial_guess_all_steep = [3.539,
                           0.450, 0.450, 0.450, 0.450,
                           1.506, 1.506, 1.506, 1.506,
                           1, 9.990,
                           0.001, 1.256]

bnds = [( 0, 0, 0, 0, 0, 0, 0, 0, 0,  1, 1,  0.001, 0.001),
        (15,15,15,15,15,15,15,15,15, 20,20,     10,    10)]

# Fitting ALL Datapoints

################
# Fit
################
result_all_soft = optimize.least_squares(non_linear_ph_model,
                                         x0 = initial_guess_all_soft,
                                         args = [data_all_soft],
                                         bounds = bnds,
                                         loss = 'linear')

#################
# Print Result
#################
# All Soft
resid_all_soft = non_linear_ph_model(result_all_soft.x, data_all_soft)
rmsd_all_soft = np.sqrt(np.sum(np.square(resid_all_soft)/len(resid_all_soft)))
print("Result for ALL Datapoints and Soft Dataset")
print("Kelec: %.3f"           % (result_all_soft.x[0] * k_b * T * np.log(10)))
print("Bp Asp: %.3f"          % (result_all_soft.x[1] * k_b * T * np.log(10)))
print("Bp Glu: %.3f"          % (result_all_soft.x[2] * k_b * T * np.log(10)))
print("Bp Lys-Arg: %.3f"      % (result_all_soft.x[3] * k_b * T * np.log(10)))
print("Bp His: %.3f"          % (result_all_soft.x[4] * k_b * T * np.log(10)))
print("Bnp Asp: %.3f"         % (result_all_soft.x[5] * k_b * T * np.log(10)))
print("Bnp Glu: %.3f"         % (result_all_soft.x[6] * k_b * T * np.log(10)))
print("Bnp Lys-Arg: %.3f"     % (result_all_soft.x[7] * k_b * T * np.log(10)))
print("Bnp His: %.3f"         % (result_all_soft.x[8] * k_b * T * np.log(10)))
print("N Max Polar: %.3f"     % (result_all_soft.x[9]))
print("N Max Non Polar: %.3f" % (result_all_soft.x[10]))
print("Alpha Polar: %.3f"     % (result_all_soft.x[11]))
print("Alpha Non Polar: %.3f" % (result_all_soft.x[12]))
print("RMSD Fit: %.3f" % (rmsd_all_soft))
print("RMSD Null: %.3f" % (rmsd_all_null))

################
# Fit
################
result_all_steep = optimize.least_squares(non_linear_ph_model,
                                          x0 = initial_guess_all_steep,
                                          args = [data_all_steep],
                                          bounds = bnds,
                                          loss = 'linear')
#################
# Print Result
#################
# All Steep
resid_all_steep = non_linear_ph_model(result_all_steep.x, data_all_steep)
rmsd_all_steep = np.sqrt(np.sum(np.square(resid_all_steep)/len(resid_all_steep)))
print("Result for ALL Datapoints and Steep Dataset")
print("Kelec: %.3f"           % (result_all_steep.x[0] * k_b * T * np.log(10)))
print("Bp Asp: %.3f"          % (result_all_steep.x[1] * k_b * T * np.log(10)))
print("Bp Glu: %.3f"          % (result_all_steep.x[2] * k_b * T * np.log(10)))
print("Bp Lys-Arg: %.3f"      % (result_all_steep.x[3] * k_b * T * np.log(10)))
print("Bp His: %.3f"          % (result_all_steep.x[4] * k_b * T * np.log(10)))
print("Bnp Asp: %.3f"         % (result_all_steep.x[5] * k_b * T * np.log(10)))
print("Bnp Glu: %.3f"         % (result_all_steep.x[6] * k_b * T * np.log(10)))
print("Bnp Lys-Arg: %.3f"     % (result_all_steep.x[7] * k_b * T * np.log(10)))
print("Bnp His: %.3f"         % (result_all_steep.x[8] * k_b * T * np.log(10)))
print("N Max Polar: %.3f"     % (result_all_steep.x[9]))
print("N Max Non Polar: %.3f" % (result_all_steep.x[10]))
print("Alpha Polar: %.3f"     % (result_all_steep.x[11]))
print("Alpha Non Polar: %.3f" % (result_all_steep.x[12]))
print("RMSD Fit: %.3f" % (rmsd_all_steep))
print("RMSD Null: %.3f" % (rmsd_all_null))

# Only IN Datapoints, range --->  s_min < ShiftExp < s_max

# Soft
data_in_soft = data_all_soft.loc[(data_all_soft['ShiftExp'] < s_max) & (data_all_soft['ShiftExp'] > s_min)]

# Steep
data_in_steep = data_all_steep.loc[(data_all_steep['ShiftExp'] < s_max) & (data_all_steep['ShiftExp'] > s_min)]

# Initial guess
initial_guess_in_soft  = [2.128,
                          0.576, 0.576, 0.576, 0.576,
                          0.000, 0.000, 0.000, 0.000,
                          3.973, 2.766,
                          0.033, 5.000]

initial_guess_in_steep = [2.128,
                          0.506, 0.506, 0.506, 0.506,
                          0.000, 0.000, 0.000, 0.000,
                          1, 1,
                          0.003, 5.910]


################
# Fit
################
result_in_soft = optimize.least_squares(non_linear_ph_model,
                                        x0 = initial_guess_in_soft,
                                        args = [data_in_soft],
                                        bounds = bnds,
                                        loss = 'linear')

#################
# Print Result
#################
# IN Soft
resid_in_soft = non_linear_ph_model(result_in_soft.x, data_in_soft)
rmsd_in_soft = np.sqrt(np.sum(np.square(resid_in_soft)/len(resid_in_soft)))
print("Result for IN Datapoints and Soft Dataset")
print("Kelec: %.3f"           % (result_in_soft.x[0] * k_b * T * np.log(10)))
print("Bp Asp: %.3f"          % (result_in_soft.x[1] * k_b * T * np.log(10)))
print("Bp Glu: %.3f"          % (result_in_soft.x[2] * k_b * T * np.log(10)))
print("Bp Lys-Arg: %.3f"      % (result_in_soft.x[3] * k_b * T * np.log(10)))
print("Bp His: %.3f"          % (result_in_soft.x[4] * k_b * T * np.log(10)))
print("Bnp Asp: %.3f"         % (result_in_soft.x[5] * k_b * T * np.log(10)))
print("Bnp Glu: %.3f"         % (result_in_soft.x[6] * k_b * T * np.log(10)))
print("Bnp Lys-Arg: %.3f"     % (result_in_soft.x[7] * k_b * T * np.log(10)))
print("Bnp His: %.3f"         % (result_in_soft.x[8] * k_b * T * np.log(10)))
print("N Max Polar: %.3f"     % (result_in_soft.x[9]))
print("N Max Non Polar: %.3f" % (result_in_soft.x[10]))
print("Alpha Polar: %.3f"     % (result_in_soft.x[11]))
print("Alpha Non Polar: %.3f" % (result_in_soft.x[12]))
print("RMSD Fit: %.3f" % (rmsd_in_soft))
print("RMSD Null: %.3f" % (rmsd_in_null))

################
# Fit
################
result_in_steep = optimize.least_squares(non_linear_ph_model,
                                         x0 = initial_guess_in_steep,
                                         args = [data_in_steep],
                                         bounds = bnds,
                                         loss = 'linear')
#################
# Print Result
#################
# IN Steep
resid_in_steep = non_linear_ph_model(result_in_steep.x, data_in_steep)
rmsd_in_steep = np.sqrt(np.sum(np.square(resid_in_steep)/len(resid_in_steep)))
print("Result for IN Datapoints and Steep Dataset")
print("Kelec: %.3f"           % (result_in_steep.x[0] * k_b * T * np.log(10)))
print("Bp Asp: %.3f"          % (result_in_steep.x[1] * k_b * T * np.log(10)))
print("Bp Glu: %.3f"          % (result_in_steep.x[2] * k_b * T * np.log(10)))
print("Bp Lys-Arg: %.3f"      % (result_in_steep.x[3] * k_b * T * np.log(10)))
print("Bp His: %.3f"          % (result_in_steep.x[4] * k_b * T * np.log(10)))
print("Bnp Asp: %.3f"         % (result_in_steep.x[5] * k_b * T * np.log(10)))
print("Bnp Glu: %.3f"         % (result_in_steep.x[6] * k_b * T * np.log(10)))
print("Bnp Lys-Arg: %.3f"     % (result_in_steep.x[7] * k_b * T * np.log(10)))
print("Bnp His: %.3f"         % (result_in_steep.x[8] * k_b * T * np.log(10)))
print("N Max Polar: %.3f"     % (result_in_steep.x[9]))
print("N Max Non Polar: %.3f" % (result_in_steep.x[10]))
print("Alpha Polar: %.3f"     % (result_in_steep.x[11]))
print("Alpha Non Polar: %.3f" % (result_in_steep.x[12]))
print("RMSD Fit: %.3f" % (rmsd_in_steep))
print("RMSD Null: %.3f" % (rmsd_in_null))

# Only OUT Datapoints, range --->  ShiftExp < s_min and ShiftExp > s_max

# Soft
data_out_soft = data_all_soft.loc[(data_all_soft['ShiftExp'] > s_max) | (data_all_soft['ShiftExp'] < s_min)]

# Steep
data_out_steep = data_all_steep.loc[(data_all_steep['ShiftExp'] > s_max) | (data_all_steep['ShiftExp'] < s_min)]

# Initial guess
initial_guess_out_soft = [10.894,
                          0.379, 0.379, 0.379, 0.379,
                          5.372, 5.372, 5.372, 5.372,
                          1, 12.166,
                          0.101, 10]

initial_guess_out_steep = [9.883,
                           3.402, 3.402, 3.402, 3.402,
                           2.784, 2.784, 2.784, 2.784,
                           11.930, 9.924,
                           5.000, 0.972]

################
# Fit
################
result_out_soft = optimize.least_squares(non_linear_ph_model,
                                         x0 = initial_guess_out_soft,
                                         args = [data_out_soft],
                                         bounds = bnds,
                                         loss = 'linear')

#################
# Print Result
#################
# OUT Soft
resid_out_soft = non_linear_ph_model(result_out_soft.x, data_out_soft)
rmsd_out_soft = np.sqrt(np.sum(np.square(resid_out_soft)/len(resid_out_soft)))
print("Result for OUT Datapoints and Soft Dataset")
print("Kelec: %.3f"           % (result_out_soft.x[0] * k_b * T * np.log(10)))
print("Bp Asp: %.3f"          % (result_out_soft.x[1] * k_b * T * np.log(10)))
print("Bp Glu: %.3f"          % (result_out_soft.x[2] * k_b * T * np.log(10)))
print("Bp Lys-Arg: %.3f"      % (result_out_soft.x[3] * k_b * T * np.log(10)))
print("Bp His: %.3f"          % (result_out_soft.x[4] * k_b * T * np.log(10)))
print("Bnp Asp: %.3f"         % (result_out_soft.x[5] * k_b * T * np.log(10)))
print("Bnp Glu: %.3f"         % (result_out_soft.x[6] * k_b * T * np.log(10)))
print("Bnp Lys-Arg: %.3f"     % (result_out_soft.x[7] * k_b * T * np.log(10)))
print("Bnp His: %.3f"         % (result_out_soft.x[8] * k_b * T * np.log(10)))
print("N Max Polar: %.3f"     % (result_out_soft.x[9]))
print("N Max Non Polar: %.3f" % (result_out_soft.x[10]))
print("Alpha Polar: %.3f"     % (result_out_soft.x[11]))
print("Alpha Non Polar: %.3f" % (result_out_soft.x[12]))
print("RMSD Fit: %.3f" % (rmsd_out_soft))
print("RMSD Null: %.3f" % (rmsd_out_null))
                                         
################
# Fit
################
result_out_steep = optimize.least_squares(non_linear_ph_model,
                                          x0 = initial_guess_out_steep,
                                          args = [data_out_steep],
                                          bounds = bnds,
                                          loss = 'linear')

#################
# Print Result
#################
# OUT Steep
resid_out_steep = non_linear_ph_model(result_out_steep.x, data_out_steep)
rmsd_out_steep = np.sqrt(np.sum(np.square(resid_out_steep)/len(resid_out_steep)))
print("Result for OUT Datapoints and Steep Dataset")
print("Kelec: %.3f"           % (result_out_steep.x[0] * k_b * T * np.log(10)))
print("Bp Asp: %.3f"          % (result_out_steep.x[1] * k_b * T * np.log(10)))
print("Bp Glu: %.3f"          % (result_out_steep.x[2] * k_b * T * np.log(10)))
print("Bp Lys-Arg: %.3f"      % (result_out_steep.x[3] * k_b * T * np.log(10)))
print("Bp His: %.3f"          % (result_out_steep.x[4] * k_b * T * np.log(10)))
print("Bnp Asp: %.3f"         % (result_out_steep.x[5] * k_b * T * np.log(10)))
print("Bnp Glu: %.3f"         % (result_out_steep.x[6] * k_b * T * np.log(10)))
print("Bnp Lys-Arg: %.3f"     % (result_out_steep.x[7] * k_b * T * np.log(10)))
print("Bnp His: %.3f"         % (result_out_steep.x[8] * k_b * T * np.log(10)))
print("N Max Polar: %.3f"     % (result_out_steep.x[9]))
print("N Max Non Polar: %.3f" % (result_out_steep.x[10]))
print("Alpha Polar: %.3f"     % (result_out_steep.x[11]))
print("Alpha Non Polar: %.3f" % (result_out_steep.x[12]))
print("RMSD Fit: %.3f" % (rmsd_out_steep))
print("RMSD Null: %.3f" % (rmsd_out_null))

# Save Results
np.savez('parameter_result_non_linear_fit.npz',
         all_soft = result_all_soft,
         all_steep = result_all_steep,
         in_soft = result_in_soft,
         in_steep = result_in_steep,
         out_soft = result_out_soft,
         out_steep = result_out_steep)

