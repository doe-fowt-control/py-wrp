import sys
sys.path.append('../py-wrp')

from src.wrp import *

# create wave gauge object and add gauges
gauges = WaveGauges()

xe = 3.12

gauges.addGauge(-3.0 - xe, 0.0648025341,   "PXI2Slot5/ai0", 0)
gauges.addGauge(-2.0 - xe, 0.06276594573,  "PXI2Slot5/ai1", 0)
gauges.addGauge(-1.4 - xe, 0.06206136076,   "PXI2Slot5/ai2", 0)
gauges.addGauge(-0.9 - xe, 0.0607484623,  "PXI2Slot5/ai3", 0)
gauges.addGauge(-0.5 - xe, 0.06026935904,   "PXI2Slot5/ai4", 0)
gauges.addGauge(-0.2 - xe, 0.06171777589,  "PXI2Slot5/ai5", 0)
gauges.addGauge(-0.0 - xe, 0.06064506793,   "PXI2Slot5/ai6", 0)
gauges.addGauge(-0, 0.06092734336,  "PXI2Slot5/ai7", 1)

times = Times(
    ta = 10,                        # reconstruction assimilation time
    ts = 30,                        # spectral assimilation time
    readRate = 40,                 # rate to read data
    writeRate = 40,                # rate to write data
    updateInterval = 1,             # interval for making a new prediction
    postWindow = 6,                 # seconds after reconstruction to visualize
    preWindow = 0,                  # seconds before data acquisition to visualize
    reconstruction_delay = 0.05,     # delay for starting write task (expected computation time)
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