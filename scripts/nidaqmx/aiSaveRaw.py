import sys
sys.path.append('../py-wrp')

from input import gauges

if __name__ == "__main__":
    from src.wrp import *
    import nidaqmx
    import numpy as np
    from nidaqmx.constants import TerminalConfiguration

    sampleRate = 100
    duration = 5
    nSamples = sampleRate * duration


    with nidaqmx.Task() as readTask:
        
        for ai_channel in gauges.portNames:
            channel = readTask.ai_channels.add_ai_voltage_chan(ai_channel)
            channel.ai_min = 0
            channel.ai_max = 5
            channel.ai_term_cfg = TerminalConfiguration.RSE


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

        np.savetxt("wallingford_test.csv", npData, delimiter=",")