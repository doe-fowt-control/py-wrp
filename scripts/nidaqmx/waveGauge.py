if __name__ == "__main__":

    sampleRate = 100
    duration = 1 # s per update
    nSamples = sampleRate * duration

    import nidaqmx
    import numpy as np
    from nidaqmx.stream_readers import (AnalogSingleChannelReader)
    import matplotlib.pyplot as plt
    from nidaqmx.constants import AcquisitionType, RegenerationMode, DigitalWidthUnits

    global plotFlag
    plotFlag = False

    with plt.ion(), nidaqmx.Task() as readTask:
        values = np.zeros(nSamples)

    # initialize plotting
        figure, ax = plt.subplots(figsize = (8,5))
        plt.ylim([0, 5])
        voltage, = ax.plot(np.linspace(0, duration, nSamples), values, color = 'blue', label = 'voltage')
        plt.title("measured voltage")
        plt.xlabel("time (s)")
        plt.ylabel("V")
        ax.legend(loc = 'upper left')
        V = figure, ax, voltage

        channel = readTask.ai_channels.add_ai_voltage_chan("PXI1Slot5/ai16")

        readTask.timing.cfg_samp_clk_timing(
            rate = sampleRate, 
            sample_mode = AcquisitionType.CONTINUOUS, 
            samps_per_chan = nSamples,
            source = 'OnboardClock',
        )

        def read_callback(task_handle, every_n_samples_event_type, number_of_samples, callback_data):
            reader.read_many_sample(
                values,
                number_of_samples_per_channel = nSamples,
            )
            global plotFlag
            plotFlag = True

            return 0
        
        reader = AnalogSingleChannelReader(readTask.in_stream)

        readTask.register_every_n_samples_acquired_into_buffer_event(
            nSamples,
            read_callback
        )

        readTask.start()


        reader.read_many_sample(
            values,
            number_of_samples_per_channel = nSamples,
        )

        while True:
            try:        
                if plotFlag:
                    # print(plotFlag)
                    voltage.set_ydata(values)
                    figure.canvas.draw()
                    figure.canvas.flush_events()
                    plotFlag = False
            except KeyboardInterrupt:
                quit()

        # data = readTask.read(
        #     number_of_samples_per_channel=nSamples,
        #     timeout = duration + 5
        # )
        # print(np.shape(data))

        # npData = np.array(data)
        # np.savetxt("resistance1.csv", npData, delimiter=",")