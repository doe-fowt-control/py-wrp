import sys
sys.path.append('../py-wrp')

from src.wrp import *

import matlab.engine
import numpy as np

# start matlab engine
eng = matlab.engine.start_matlab()

# change engine directory to src where matlab files are
eng.cd(r'scripts/matlab-engine/src', nargout = 0)

# use standard timing definitions
times = Times(
    ta = 10,                        # reconstruction assimilation time
    ts = 30,                        # spectral assimilation time
    readRate = 60,                 # rate to read data
    writeRate = 20,                # rate to write data
    updateInterval = 1,             # interval for making a new prediction
    postWindow = 6,                 # seconds after reconstruction to visualize
    preWindow = 0,                  # seconds before data acquisition to visualize
    reconstruction_delay = 0.05,     # delay for starting write task (expected computation time)
)

(float_excitation_data, 
 parameters_cell_array, 
 onlineData, 
 ekf,
 y,
 x, 
 controller_rack_length, 
 n_x, 
 n_y, 
 n_md, 
 nlobj) = eng.initializeController(float(5), 1, 0.05, nargout=11)

# mv = matlab.double([controller_rack_length / 2])
mv = controller_rack_length / 2
# b = matlab.double([11, 22, 33])
u_ekf = matlab.double([mv, 0, 0, 0])

hp = 5      # number of samples in prediction horizon
prediction_horizon_times = matlab.double([(np.arange(0, hp)/times.writeRate).tolist()])

# predicted_amplitude = matlab.double(np.zeros((1, 100)).tolist())
# predicted_ang_freq = matlab.double(np.zeros((1, 100)).tolist())
# predicted_phase = matlab.double(np.zeros((1, 100)).tolist())
# predicted_preview_wave_elevation = matlab.double(np.zeros((1, 5)).tolist())

predicted_amplitude = matlab.double([0]*100)
predicted_ang_freq = matlab.double([0]*100)
predicted_phase = matlab.double([0]*100)
predicted_preview_wave_elevation = matlab.double([0]*hp, size=[hp,1])

(mv, 
 u_ekf,
 onlineData,
 xk,
 md_prediction) = eng.updateController(
    ekf,        # EKF matlab object from initialize controller
    y,
    u_ekf,
    parameters_cell_array,
    float_excitation_data,
    prediction_horizon_times,
    predicted_amplitude,
    predicted_ang_freq,
    predicted_phase,
    predicted_preview_wave_elevation,
    mv,
    onlineData, 
    nargout = 5
)

# # t[0]
# AQWA = t[0]
# AQWA['AQWA_frequencies_scaled']
# AQWA['excitation_amp_heave']
# AQWA['excitation_phase_heave']
# AQWA['excitation_amp_roll']
# AQWA['excitation_phase_roll']

# # t[1]
# gravity = t[1][0]
# density_water = t[1][1]
# total_mass_heave = t[1][2]
# total_inertia_roll = t[1][3]
# beam_model = t[1][4]
# operating_draft = t[1][5]
# equivalent_box_length = t[1][6]
# C_44 = t[1][7]
# C_34 = t[1][8]
# viscous_drag_factor_heave = t[1][9]
# viscous_drag_factor_roll = t[1][10]
# number_prony_heave = t[1][11]
# beta_heave_real = t[1][12]
# s_heave_real = t[1][13]
# beta_heave_imag = t[1][14]
# s_heave_imag = t[1][15]
# number_prony_roll = t[1][16]
# beta_roll_real = t[1][17]
# s_roll_real = t[1][18]
# beta_roll_imag = t[1][19]
# s_roll_imag = t[1][20]
# I_Re_3_start = t[1][21]
# I_Re_3_end = t[1][22]
# I_Im_3_start = t[1][23]
# I_Im_3_end = t[1][24]
# I_Re_4_start = t[1][25]
# I_Re_4_end = t[1][26]
# I_Im_4_start = t[1][27]
# I_Im_4_end = t[1][28]
# control_on_flag = t[1][29]
# scale = t[1][30]
# scaled_displacement = t[1][31]
# control_rack_length = t[1][32]
# control_mass = t[1][33]
# controller_sampling_time = t[1][34]

# print(t[3])
# t[2]['ref']
# print(t[3])
# print(t[4])
# print(t[5])
# print(t[6])
# print(t[7])
# print(t[8])
# print(t[9])
# print(t[10])
