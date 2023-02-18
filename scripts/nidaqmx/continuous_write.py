import sys
sys.path.append('../py-wrp')

# has a good approach to the read callback using bufferUpdate method
# write callback looks a little broken

if __name__ == "__main__":
    from src.wrp import *
    import matplotlib.pyplot as plt
    import numpy as np


    import nidaqmx
    from nidaqmx.stream_writers import (AnalogMultiChannelWriter)
    from nidaqmx.constants import AcquisitionType, RegenerationMode, DigitalWidthUnits

    # initialize parameters with default settings
    pram = Params()

    # create wave gauge object and add gauges
    gauges = WaveGauges()
    			
    gauges.addGauge(-7, 0.1105386774, "PXI1Slot5/ai6", 0)
    gauges.addGauge(-5, 0.08484144985, "PXI1Slot5/ai0", 0)
    gauges.addGauge(-4, 0.1119472234, "PXI1Slot5/ai4", 0)
    gauges.addGauge(-0, 0.08497254516, "PXI1Slot5/ai2", 1)

    # create flow object which manages transferring data to wrp
    flow = DataManager(
        pram, 
        gauges,
        readSampleRate = 20,
        writeSampleRate = 40,
        updateInterval = 2
    )

    # set up the wrp
    wrp = WRP(gauges)

    # def plotFunc(V, newData):
    #     '''Update data for an existing set of matplotlib objects
    #     '''
    #     figure, ax, line = V
    #     line.set_ydata(newData)
    #     figure.canvas.draw()
    #     figure.canvas.flush_events()

    with plt.ion(), nidaqmx.Task() as writeTask:

# PLOTS
        # global V
        # # initialize plotter
        # V = wrp.setVis(flow)

# PORTS + TIMING
            # write task
        writeTask.ao_channels.add_ao_voltage_chan("PXI1Slot5/ao2")
        # writeTask.ao_channels.add_ao_voltage_chan("PXI1Slot4/ao0")

        writeTask.timing.cfg_samp_clk_timing(
            rate = flow.writeSampleRate, 
            sample_mode = AcquisitionType.CONTINUOUS, 
            samps_per_chan = flow.writeNSamples,
            source = 'OnboardClock',
        )

        # writeTask.ao_data
        # require the write task to ask for new data, rather than simply repeating
        writeTask.out_stream.regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION




    # DEFINE CALLBACK
        def write_callback(task_handle, every_n_samples_event_type, number_of_samples, callback_data):
            # print('write')
        # select the last readNSamples of the specified column (gauge) in the stored array
            # readLatest = flow.bufferValues[3, -flow.readNSamples:]
        # create interpolated function from measured series
            # f = interpolate.interp1d(readLatest, flow.readTime, fill_value="extrapolate")
        # evaluate function at necessary intervals
            # writeLatest = f(flow.writeTime)

            # writeLatest = readLatest

            # writeLatest[writeLatest < 0] = 0
            # writeLatest[writeLatest > 10] = 0

            
            voltageScale = 19.23
            A = 0.0254 * 3.5 * voltageScale # m
            # Sinusoidal input
            writeVals = A*np.sin(3.14159*flow.writeTime) + 2.5
            writeLatest = np.reshape(writeVals, (1,flow.writeNSamples))
            
            # # Static input
            # writeLatest = A*np.ones((1, flow.writeNSamples)) + 2.5

            writer.write_many_sample(writeLatest)
            return 0
        
    # STREAMS
        writer = AnalogMultiChannelWriter( # single channel
            writeTask.out_stream, 
            auto_start = False)

    # REGISTER CALLBACK
        writeTask.register_every_n_samples_transferred_from_buffer_event(
            flow.writeNSamples,
            write_callback)

    # START
        writer.write_many_sample(flow.writeValues)
        # write task has to start before read so that it can detect the trigger
        writeTask.start()

        
        


# update the plot
        # while True:
        #     try:
        #         if wrp.plotFlag:
        #             wrp.updateVis(flow, V)
        #     except KeyboardInterrupt:
        #         quit()

# use input to keep collection going
        input('press ENTER to stop')

