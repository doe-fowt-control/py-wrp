import sys
sys.path.append('../py-wrp')


from src.wrp import Params, WaveGauges, DataLoader, DataManager, WRP
import matplotlib.pyplot as plt
import numpy as np
import time

if __name__ == "__main__":
    # initialize parameters with default settings
    pram = Params()

    # create wave gauge object and add gauges
    gauges = WaveGauges()

    xi = 16.17
    gauges.addGauge(11.88 - xi, 1, "-", 0)
    gauges.addGauge(12.32 - xi, 1, "-", 0)
    gauges.addGauge(12.72 - xi, 1, "-", 0)
    gauges.addGauge(0, 1, "-", 1)

    # gauges.addGauge(-4, .08083, "PXI1Slot5/ai2", 0)
    # gauges.addGauge(-3.5, .10301, "PXI1Slot5/ai6", 0)
    # gauges.addGauge(-2, .10570, "PXI1Slot5/ai4", 0)
    # gauges.addGauge(-0, .08163, "PXI1Slot5/ai0", 1) # the '1' here indicates for prediction

    # create dm object which manages transferring data to wrp
    dm = DataManager(
        pram,
        gauges,
        readSampleRate = 100, # this is dictated by the rate at which data was originally collected
        writeSampleRate = 100,
        updateInterval = 1,
    )

# files to read statically
    load = DataLoader(
        'data/reg_eta_stephanie.csv',
        'data/reg_t_stephanie.csv',
        # 'data/3.12.22.full.csv',
        # 'data/3.12.22.time.csv',
    )


    # initialize wrp
    wrp = WRP(gauges)

    eta_measured = []
    eta_predicted = []
    global aphio
    aphio = np.zeros((3, 100))

    # initialize plotter
    V = wrp.setVis(dm)

    # specify operation to be triggered whenever 'loading' data
    def callFunc(wrp, dm):
        global V
        global aphio
        # start_time = time.time()
        wrp.spectral(dm)
        wrp.inversion(dm)
        wrp.reconstruct(dm)

        try:
            eta_measured.extend(load.newMeasurement)
            eta_predicted.extend(wrp.reconstructedSurfacePredict)
            
            new_aphio = np.transpose(np.hstack((wrp.A, wrp.phi, wrp.w)))
            aphio = np.hstack((aphio, new_aphio))

        except:
            pass
        
        # print(np.shape(wrp.A))
        # print(np.shape(wrp.phi))
        # print(np.shape(wrp.w))

        # print(np.shape(np.transpose(np.hstack((wrp.A, wrp.phi, wrp.w)))))



        # print(time.time() - start_time)
        # wrp.updateVis(dm, V)

        # plt.pause(.1)

    with plt.ion():
        load.generateBuffersDynamic(dm, wrp, 60, callFunc)

    true_pred = np.vstack((eta_measured, eta_predicted))

    print(np.shape(aphio))
    print(np.shape(true_pred))

# trim first big chunk and add the padding back in
    aphio = aphio[:, 2100:]
    aphio = np.hstack((np.zeros((3,100)), aphio))
# trim a corresponding chunk
    true_pred = true_pred[:, 2000:]
    print(np.shape(aphio))
    print(np.shape(true_pred))

    # np.savetxt("reg_A_phi_omega.csv", aphio, delimiter=',')
    # np.savetxt("reg_eta_true_predicted.csv", true_pred, delimiter = ",")

    # print(np.shape(aphio))
    # print(np.shape(eta_predicted))
    # print(np.shape(eta_measured))

    # print(np.shape(dm.validateData()))
    # plt.plot(eta_predicted)
    # plt.plot(dm.validateData())
    # plt.show()