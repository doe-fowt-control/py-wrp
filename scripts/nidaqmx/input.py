import sys
sys.path.append('../py-wrp')

from src.wrp import *

# create wave gauge object and add gauges
gauges = WaveGauges()
            
gauges.addGauge(-7, 0.1105386774,   "PXI2Slot5/ai0", 0)
gauges.addGauge(-5, 0.08484144985,  "PXI2Slot5/ai1", 0)
gauges.addGauge(-4, 0.1119472234,   "PXI2Slot5/ai2", 0)
gauges.addGauge(-0, 0.08497254516,  "PXI2Slot5/ai3", 1)

times = Times(
    ta = 5,                        # reconstruction assimilation time
    ts = 10,                        # spectral assimilation time
    readRate = 40,                 # rate to read data
    writeRate = 40,                # rate to write data
    updateInterval = 1.25,             # interval for making a new prediction
    postWindow = 6,                 # seconds after reconstruction to visualize
    preWindow = 0,                  # seconds before data acquisition to visualize
    reconstruction_delay = 0.5,     # delay for starting write task (expected computation time)
)

prams = Prams(
    nf = 100,                       # number of harmonics to use in reconstruction
    mu = 0.05,                      # threshold energy cutoff for prediction zone
    lam = 1,                        # regularization parameter for simple inversion technique
)

# create dm object which manages transferring data to wrp
dm = DataManager(
    prams, 
    gauges,
    times
)

# initialize wave reconstruction code which handles the math
wrp = WRP(
    prams, 
    gauges,
    times
)