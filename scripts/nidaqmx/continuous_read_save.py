import sys
sys.path.append('../py-wrp')

# has a good approach to the read callback using bufferUpdate method
# write callback looks a little broken

if __name__ == "__main__":
    from src.wrp import *
    import matplotlib.pyplot as plt
    import numpy as np


    import nidaqmx
    from nidaqmx.stream_readers import (AnalogMultiChannelReader)
    from nidaqmx.stream_writers import (AnalogMultiChannelWriter)
    from nidaqmx.constants import AcquisitionType, RegenerationMode, DigitalWidthUnits

    # initialize parameters with default settings
    pram = Params()

    # create wave gauge object and add gauges
    gauges = WaveGauges()


    gauges.addGauge(0, 1, "Dev1/ai3", 0)
    gauges.addGauge(0, 1, "Dev1/ai4", 0)

    # create flow object which manages transferring data to wrp
    flow = DataManager(
        pram, 
        gauges,
        readSampleRate = 25,
        writeSampleRate = 100,
        updateInterval = 1
    )

    # set up the wrp
    wrp = WRP(gauges)


    with plt.ion(), nidaqmx.Task() as readTask:

# PLOTS
        # global V
        # # initialize plotter
        # V = wrp.setVis(flow)

# PORTS + TIMING
        # read task
        for ai_channel in gauges.portNames:
            readTask.ai_channels.add_ai_voltage_chan(ai_channel)

        readTask.timing.cfg_samp_clk_timing(
            rate = flow.readSampleRate, 
            sample_mode = AcquisitionType.CONTINUOUS, 
            samps_per_chan = flow.readNSamples,
            source = 'OnboardClock',
        )
        global m
        m = np.zeros((2, 1))
    # DEFINE CALLBACK
        def read_callback(task_handle, every_n_samples_event_type, number_of_samples, callback_data):
            # print('read')
            global m
        # read new data into readValues
            reader.read_many_sample(
                flow.readValues,
                number_of_samples_per_channel = flow.readNSamples,
            )
            m = np.hstack((m, flow.readValues))
            return 0

        
    # STREAMS
        reader = AnalogMultiChannelReader( # multi channel
            readTask.in_stream)


    # REGISTER CALLBACK
        readTask.register_every_n_samples_acquired_into_buffer_event(
            flow.readNSamples,
            read_callback)


    # START

        readTask.start()
        
        
        reader.read_many_sample(
            flow.readValues,
            number_of_samples_per_channel = flow.readNSamples,
        )

# update the plot
        # while True:
        #     try:
        #         if wrp.plotFlag:
        #             wrp.updateVis(flow, V)
        #     except KeyboardInterrupt:
        #         quit()

# use input to keep collection going
        input('press ENTER to stop collection')
        # print(m)
        np.savetxt("foo.csv", m, delimiter=",")

