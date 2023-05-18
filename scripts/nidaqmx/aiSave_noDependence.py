

if __name__ == "__main__":
    import nidaqmx
    import numpy as np
    from nidaqmx.constants import TerminalConfiguration
    import matplotlib.pyplot as plt

    sampleRate = 1000
    duration = 10
    nSamples = sampleRate * duration

    # portNames = [
    #     "PXI1Slot5/ai0",
    #     "PXI1Slot5/ai1",
    #     "PXI1Slot5/ai2",
    #     "PXI1Slot5/ai3",
    #     "PXI1Slot5/ai4",
    #     "PXI1Slot5/ai5",
    #     "PXI1Slot5/ai6",
    #     "PXI1Slot5/ai7",
    # ]
    portNames = [
        "PXI1Slot7/ai0",
        "PXI1Slot7/ai1",
    ]

    with nidaqmx.Task() as readTask:
        
        for ai_channel in portNames:
            channel = readTask.ai_channels.add_ai_voltage_chan(ai_channel)
            channel.ai_min = 0
            channel.ai_max = 5
            channel.ai_term_cfg = TerminalConfiguration.DIFF


        readTask.timing.cfg_samp_clk_timing(
            rate = sampleRate, 
            source = "OnboardClock",
            samps_per_chan = nSamples
        )

        # in_stream = readTask.in_stream
        # readTask.start()
        data = readTask.read(
            number_of_samples_per_channel=nSamples,
            timeout = duration + 5
        )

        # print(np.shape(data))

        npData = np.array(data)
        

        print(np.mean(npData[:, -1000:], axis = 1))
        # plt.plot(npData[0, :])
        # plt.show()
        np.savetxt("D:\saveVals.csv", npData, delimiter=",")
