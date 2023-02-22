import sys
sys.path.append('../py-wrp')

# gauges are configured in separate input file
from input import times, gauges, dm, wrp

# has a good approach to the read callback using bufferUpdate method
# write callback looks a little broken

if __name__ == "__main__":
    from src.wrp import *
    import matplotlib.pyplot as plt
    import numpy as np


    import nidaqmx
    from nidaqmx.stream_readers import (AnalogMultiChannelReader)
    from nidaqmx.stream_writers import (AnalogMultiChannelWriter)
    from nidaqmx.constants import AcquisitionType, RegenerationMode, DigitalWidthUnits, TerminalConfiguration

    # def plotFunc(V, newData):
    #     '''Update data for an existing set of matplotlib objects
    #     '''
    #     figure, ax, line = V
    #     line.set_ydata(newData)
    #     figure.canvas.draw()
    #     figure.canvas.flush_events()

    with plt.ion(), nidaqmx.Task() as readTask, nidaqmx.Task() as writeTask:

# PLOTS
        global V
        # initialize plotter
        V = wrp.setVis(dm)

# PORTS + TIMING
        # read task
        for ai_channel in gauges.portNames:
            channel = readTask.ai_channels.add_ai_voltage_chan(ai_channel)
            channel.ai_min = 0
            channel.ai_max = 5
            channel.ai_term_cfg = TerminalConfiguration.RSE
        
        readTask.timing.cfg_samp_clk_timing(
            rate = dm.readRate, 
            sample_mode = AcquisitionType.CONTINUOUS, 
            samps_per_chan = dm.read.nSamples,
            source = 'OnboardClock',
        )

            # write task
        writeTask.ao_channels.add_ao_voltage_chan("PXI2Slot5/ao0")

        writeTask.timing.cfg_samp_clk_timing(
            rate = dm.writeRate, 
            sample_mode = AcquisitionType.CONTINUOUS, 
            samps_per_chan = dm.write.nSamples,
            source = 'OnboardClock',
        )

# writeTask trigger
        writeTask.triggers.start_trigger.cfg_dig_edge_start_trig(
            trigger_source=readTask.triggers.start_trigger.term,
        )
        writeTask.triggers.start_trigger.delay_units = DigitalWidthUnits.SECONDS
        writeTask.triggers.start_trigger.delay = times.reconstruction_delay

        # require the write task to ask for new data, rather than simply repeating
        writeTask.out_stream.regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION


    # DEFINE CALLBACK
        def read_callback(task_handle, every_n_samples_event_type, number_of_samples, callback_data):
            # print('read')

        # read new data into read.values
            reader.read_many_sample(
                dm.read.values,
                number_of_samples_per_channel = dm.read.nSamples,
            )

        # add new data to buffer
            dm.bufferUpdate(dm.read.values)

            # wrp.icwm(dm)
            wrp.lwt(dm) # here's another option. different methods in this 'full' script for different wave models

            return 0

        def write_callback(task_handle, every_n_samples_event_type, number_of_samples, callback_data):
            # print('write')
            
            try:
                writeLatest = np.reshape(wrp.reconstructedSurfacePredict, (1, dm.write.nSamples))

            except AttributeError:
                voltageScale = 19.23
                A = 0.0254 * 3.5 * voltageScale # m
                writeVals = A*np.sin(np.pi/5*dm.write.time) + 2.5 
                writeLatest = np.reshape(writeVals, (1,dm.write.nSamples))
                print('default')
            
            writer.write_many_sample(writeLatest)


            return 0
        
    # STREAMS
        reader = AnalogMultiChannelReader( # multi channel
            readTask.in_stream)
        writer = AnalogMultiChannelWriter( # single channel
            writeTask.out_stream, 
            auto_start = False)

    # REGISTER CALLBACK
        readTask.register_every_n_samples_acquired_into_buffer_event(
            dm.read.nSamples,
            read_callback)

        writeTask.register_every_n_samples_transferred_from_buffer_event(
            dm.write.nSamples,
            write_callback)

    # START
        writer.write_many_sample(dm.write.values)
        # write task has to start before read so that it can detect the trigger
        writeTask.start()
        readTask.start()
        
        
        reader.read_many_sample(
            dm.read.values,
            number_of_samples_per_channel = dm.read.nSamples,
        )

# update the plot
        while True:
            try:
                # if wrp.plotFlag:
                wrp.updateVis(dm, V)
            except KeyboardInterrupt:
                quit()

# # use input to keep collection going
#         input('press ENTER to stop')

