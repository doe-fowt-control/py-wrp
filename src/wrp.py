import numpy as np
from scipy.signal import welch
from scipy import linalg
import matplotlib.pyplot as plt
from numpy import genfromtxt
import time


class WaveGauges:
    """The information needed to a interpret a physical wave measurement
    
    Contains lists with the physical locations, calibration constants, analog
    port, and role in the WRP algorithm, which can be either `measurement` or 
    `validation`. The latter also indicates the physical location of the float.
    Position is defined where waves move in positive x direction and x = 0 at the
    validation gauge.

    Attributes:
        xPositions: list of physical locations
        calibrationSlopes: list of conversion factors from measurement to wave height, (m/V)
        portNames: list of channels on which to aquire data from this wave gauge
        wrpRole: list of roles, 0 indicates measurement, 1 indicates validation
    """
    def __init__(self):
        """Set up lists for pertinent wave gauge information
        """
        self.xPositions = []
        self.calibrationSlopes = []
        self.portNames = []
        self.wrpRole = []      # 0 = measurement gauge, 1 = prediction gauge

    def addGauge(self, position, slope, name, role):
        """Adds details for an add gauge to class

        Args:
            position: physical location in space, meters
            slope: conversion factor for measurement m/V
            name: analog channel/ address, string
            role: 0 for measurement, 1 for validation

        """
        self.xPositions.append(position)
        self.calibrationSlopes.append(slope)
        self.portNames.append(name)
        self.wrpRole.append(role)

    def nGauges(self):
        """Determine number of gauges which have been added"""
        return(len(self.xPositions))

    def measurementIndex(self):
        """Find indices of gauges used for measurement
        
        Returns:
            A list of indices
        """
        mg = [i for i, e in enumerate(self.wrpRole) if e == 0]
        return mg

    def predictionIndex(self):
        """Find indices of gauges used for validation
        
        Returns:
            A list of indices
        """
        pg = [i for i, e in enumerate(self.wrpRole) if e != 0]
        return pg

class Prams:
    """Parameters which alter the inversion and reconstruction process
    
    Attributes:
        nf: number of frequencies to use for reconstruction
        mu: threshold parameter to determine fastest/slowest 
            group velocities for prediction zone
        lam: regularization parameter for least squares fit
    """
    def __init__(
        self,
        nf = 100,
        mu = 0.05,
        lam = 1,
    ):
        self.nf = nf        # number of harmonics in reconstruction
        self.mu = mu        # cutoff threshold for prediction zone
        self.lam = lam      # regularization in inverse problem

class Times:
    """Timing information for the WRP and data acquisition process
    
    Attributes:
        ta: reconstruction assimilation time
        ts: spectral assimilation time
        readRate: frequency to read
        writeRate: frequency to write
        updateInterval: spacing between callback
        postWindow: validation reconstruction after reconstruction time
        preWindow: validation reconstruction before reconstruction time
        reconstruction_delay: delay between read and write processes
    """
    def __init__(
        self,
        ta = 15,
        ts = 30,
        readRate = 100,
        writeRate = 100,
        updateInterval = 1,
        postWindow = 6,
        preWindow = 0,
        reconstruction_delay = 0.5,
    ):
        
        if postWindow % updateInterval != 0: # round up window to nearest multiple
            postWindow = postWindow + (updateInterval - (postWindow % updateInterval))
        

        self.ta = ta            # 
        self.ts = ts            # 
        self.readRate = readRate
        self.writeRate = writeRate
        self.updateInterval = updateInterval
        self.postWindow = postWindow
        self.preWindow = preWindow
        self.reconstruction_delay = reconstruction_delay

class static:
    """static arrays and their corresponding time series"""
    def __init__(self, rate, t_start, t_end, nRows):
        dt = 1/rate
        duration = t_end - t_start

        # set up read - samples, values, time -
        self.nSamples = int(rate * duration) # int to account for non integer update intervals
        self.values = np.zeros((nRows, self.nSamples), dtype=np.float64)
        self.time = np.arange(t_start, t_end, dt)

class DataManager:
    """Facilitates data allocation to wrp and control
    
    Args:
        pram: Params instance
        gauges: WaveGauges instance
        readRate: frequency to read
        writeRate: frequency to write
        updateInterval: spacing between callback

    
    """
        # voltageScale = 19.23
        # A = 0*0.0254 * 3.5 * voltageScale # m
        # writeVals = (A*np.sin(3.14159*self.write.time)) + 2.5

    def __init__(self, pram, gauges, time):

        # pull from time
        self.readRate = time.readRate  # frequency to take wave measurements (Hz)
        self.writeRate = time.writeRate # frequency to send motor commands (Hz)
        self.updateInterval = time.updateInterval     # time between grabs at new data (s)
        reconstruction_delay = time.reconstruction_delay
        
        # number of samples for reconstruction
        self.assimilationSamples = time.ta * self.readRate

        # time interval between wave measurements (s)
        self.readDT = 1 / self.readRate
        self.writeDT = 1 / self.writeRate

        # pull from gauges
        self.nChannels = gauges.nGauges()       # number of channels from which to read
        self.mg = gauges.measurementIndex()     # gauges to select for reconstruction
        self.pg = gauges.predictionIndex()      # gauges for prediction
        self.calibrationSlopes = np.expand_dims(gauges.calibrationSlopes, axis = 1)  # alter calibration constants for easy multiplying

        self.read = static(
            self.readRate,
            0,
            self.updateInterval,
            self.nChannels,
        )

        self.write = static(
            self.writeRate,
            reconstruction_delay,
            self.updateInterval + reconstruction_delay,
            1,
        )

        self.buffer = static(
            self.readRate,
            -time.ts,
            0,
            self.nChannels,
        )

        self.validateWrite = static(
            self.writeRate,
            -time.ta - time.preWindow,
            time.postWindow,
            self.nChannels
        )

        self.validateRead = static(
            self.readRate,
            -time.ta - time.preWindow,
            time.postWindow,
            self.nChannels
        )

        # array to save results of inversions for the length of time to visualize in the future
        self.inversionNSaved = int(time.postWindow / self.updateInterval) + 1
        self.inversionSavedValues = np.zeros((2, self.inversionNSaved, pram.nf))



    def bufferUpdate(self, newData):
        """adds new data to the end of buffer.values, shifting existing data and removing the oldest
        
        Args:
            newData: array of new data collected with length read.nSamples
        """
        # shift old data to the end of the matrix
        self.buffer.values = np.roll(self.buffer.values, -self.read.nSamples)
        # write over old data with new data
        self.buffer.values[:, -self.read.nSamples:] = newData

    def inversionUpdate(self, a, b):
        """adds most recent inversion to the end ofinversionSavedValues, deletes the oldest
        
        Args:
            a: array of weights for cosine
            b: array of weights for sine
        """
        # array to save backlog of inversion results, good for validating real time
        self.inversionSavedValues = np.roll(self.inversionSavedValues, -1, axis = 1)

        # need to squeeze to fit it into the matrix
        self.inversionSavedValues[0][self.inversionNSaved - 1] = np.squeeze(a)
        self.inversionSavedValues[1][self.inversionNSaved - 1] = np.squeeze(b)


    def inversionGetValues(self, method):
        """retrieves inversion values for a specified `method`
        
        Args: 
            method: string
                'validate' -> values for validation
                'predict' -> most recent values for prediction
        
        Returns:
            a: array of weights for cosine
            b: array of weights for sine
        """
        # need expand_dims for the matrix math in reconstruct

        if method == 'validate':
            a = np.expand_dims(self.inversionSavedValues[0][0][:], axis=1)
            b = np.expand_dims(self.inversionSavedValues[1][0][:], axis=1)

            return a,b

        if method == 'predict':
            a = np.expand_dims(self.inversionSavedValues[0][-1][:], axis=1)
            b = np.expand_dims(self.inversionSavedValues[1][-1][:], axis=1)

            return a,b

    def reconstructionData(self):
        """gets data from buffer.values which should be used for reconstruction
        
        Calls preprocessing to scale and center data on mean

        Returns:
            An array of processed data for reconstruction
        """
        # select measurement gauges across reconstruction time
        data = self.buffer.values[self.mg, -self.assimilationSamples:]

        processedData = self.preprocess(data, self.mg)
        return processedData

    def reconstructionTime(self):
        """gets time to use for reconstruction
        
        Returns:
            Array of time values for reconstruction
        """
        time = self.buffer.time[-self.assimilationSamples:]
        return time

    def spectralData(self):
        """gets data from buffer.values which should be used as spectral data
        
        Calls preprocessing to scale and center data on mean

        Returns:
            An array of processed data for spectral information
        """
        data = self.buffer.values[self.mg, :]

        processedData = self.preprocess(data, self.mg)
        return processedData

    def validateData(self):
        """selects and scales validation data from local buffer
        
        Returns:
            processed data for validation
        """

        data = self.buffer.values[
            self.pg,
            -self.validateRead.nSamples:,
        ]

        processedData = self.preprocess(data, self.pg)
        return processedData

    def preprocess(self, data, whichGauges):
        """scales data by calibration constants and subtracts the mean
        
        Args: 
            data: array of values to be processed
            whichGauges: the indices of the gauges which correspond to the data being processed
        
        Returns: 
            array of processed data
        """
        # scale by calibration constants
        data *= self.calibrationSlopes[whichGauges]

        # center on mean
        dataMeans = np.expand_dims(np.mean(data, axis = 1), axis = 1)
        data -= dataMeans

        return data



class WRP:
    """Implements methods of wave reconstruction and propagation

    Args: 
        gauges: instance of WaveGauges
    """
    def __init__(self, pram, gauges, time):
        self.mu = pram.mu
        self.lam = pram.lam
        self.nf = pram.nf

        self.ta = time.ta
        self.ts = time.ts


        self.x = gauges.xPositions
        self.calibration = gauges.calibrationSlopes

        # gauges to select for reconstruction
        self.mg = gauges.measurementIndex()

        # gauges for prediction
        self.pg = gauges.predictionIndex()

        # important spatial parameters for wrp based on gauge locations
        self.xmax = np.max( np.array(self.x)[self.mg] )
        self.xmin = np.min( np.array(self.x)[self.mg] )
        self.xpred = np.array(self.x)[self.pg]

        self.plotFlag = False

    def spectral(self, dm):
        """Calculates spectral information

        Uses spectral data to create a set of spectral attributes including \n
            - T_p: peak period
            - k_p: peak wavenumber
            - m0: zero moment of the spectrum
            - Hs: significant wave height
            - cg_fast, cg_slow: fastest and slowest group velocity
            - xe, xb: spatial reconstruction parameters
            - k_min, k_max: wavenumber bandwidth for reconstruction

        Args:
            dm: instance of DataManager
        """
        # assign spectral variables to wrp class
        data = dm.spectralData()

        # check to see if the buffer is filled
        self.bufferFilled = data[0][0] != data[0][1]
        print(self.bufferFilled)


        if self.bufferFilled:
            f, pxxEach = welch(data, fs = dm.readRate)
            pxx = np.mean(pxxEach, 0)
    

            self.T_p = 1 / (f[pxx == np.max(pxx)])

            # peak wavelength
            self.k_p = (1 / 9.81) * (2 * np.pi / self.T_p)**2

            # zero-th moment as area under power curve
            self.m0 = np.trapz(pxx, f)

            # significant wave height from zero moment
            self.Hs = 4 * np.sqrt(self.m0)

            self.w = f * np.pi * 2

            thresh = self.mu * np.max(pxx)

            # set anything above the threshold to zero
            pxx[pxx > thresh] = 0

            # print(np.shape(pxx))
            # plt.plot(f, pxx)
            # find the locations which didn't make the cut
            pxxIndex = np.nonzero(pxx)[0]

            # find the largest gap between nonzero values
            low_index = np.argwhere( (np.diff(pxxIndex) == np.max(np.diff(pxxIndex))) )[0][0]
            high_index = pxxIndex[low_index + 1]

            # plt.axvline(x = f[low_index])
            # plt.axvline(x = f[high_index])
            # plt.show()

            # select group velocities
            if self.w[low_index] == 0:
                self.cg_fast = 20 # super arbitrary
            else:
                self.cg_fast = (9.81 / (self.w[low_index] * 2))
            self.cg_slow = (9.81 / (self.w[high_index] * 2))

            # spatial parameters for reconstruction bandwidth
            self.xe = self.xmax + self.ta * self.cg_slow
            self.xb = self.xmin

            # reconstruction bandwidth wavenumbers
            self.k_min = 2 * np.pi / (self.xe - self.xb)
            self.k_max = 2 * np.pi / min(abs(np.diff(self.x)))


    def lwt(self, dm):
        """Runs complete set of in-the-loop operations for linear wave theory
        
        # this single call to the icwm method does a bunch of stuff
        # - evaluation of the spectrum in its current state
        # - inversion at the current time step, save inversion values
        # - reconstruct at the time interval needed for the control system
        # - reconstruct an old time series for the validation plot
        """
        self.spectral(dm)
        # print(self.bufferFilled)
        # print(dm.buffer.values)

        # only do the actions once enough data is available to evaluate the spectrum
        if self.bufferFilled:
            self.inversion_lwt(dm)
            self.reconstruct_lwt(dm, 'validate')
            self.reconstruct_lwt(dm, 'predict')


    def inversion_lwt(self, dm):
        """Find linear weights for surface representation

        Calculates an array of wavenumbers and corresponding deep water
        frequencies. Solves least squares optimization to get best fit
        surface representation. Adds results of inversion to a saved 
        array in DataManager called inversionSavedValues.

        Args:
            dm: instance of DataManager
        """
    # define wavenumber and frequency range
        k = np.linspace(self.k_min, self.k_max, self.nf)
        w = np.sqrt(9.81 * k)


    # get data
        eta = dm.reconstructionData()
        t = dm.reconstructionTime()
        x = np.array(self.x)[self.mg]

    # grid data and reshape for matrix operations
        X, T = np.meshgrid(x, t)

        self.k = np.reshape(k, (self.nf, 1))
        self.w = np.reshape(w, (self.nf, 1))

        X = np.reshape(X, (1, np.size(X)), order='F')

        T = np.reshape(T, (1, np.size(T)), order='F')        
        eta = np.reshape(eta, (np.size(eta), 1))

        psi = np.transpose(self.k@X - self.w@T)

        
    # data matrix
        Z = np.zeros((np.size(X), 2*self.nf))
        Z[:, :self.nf] = np.cos(psi)
        Z[:, self.nf:] = np.sin(psi)


        m = np.transpose(Z)@Z + (np.identity(self.nf * 2) * self.lam)
        n = np.transpose(Z)@eta
        weights, res, rnk, s = linalg.lstsq(m, n)

        # choose all columns [:] for future matrix math
        a = weights[:self.nf,:]
        b = weights[self.nf:,:]


        self.A = np.sqrt(a**2 + b**2)
        self.phi = np.arctan2(b,a)

        dm.inversionUpdate(a, b)


    def reconstruct_lwt(self, dm, intent):
        """Reconstructs surface using saved inversion values
        
        Calculates upper and lower limit time boundary for reconstruction time.
        Calculates shape of reconstructed surface for both validation and 
        prediction and saves them as attributes of DataManager. The former 
        is saved as DataManager.reconstructedSurfaceValidate and the latter as
        DataManager.reconstructedSurfacePredict

        Args:
            dm: instance of DataManager
        """
# General
        # prediction zone time boundary
        self.t_min = (1 / self.cg_slow) * (self.xpred - self.xe)
        self.t_max = (1 / self.cg_fast) * (self.xpred - self.xb)

        # matrix for summing across frequencies
        sumMatrix = np.ones((1, self.nf))

        if intent == 'validate':
            t = np.expand_dims(dm.validateWrite.time, axis = 0)
            dx = self.xpred * np.ones((1, dm.validateWrite.nSamples))
            
            a, b = dm.inversionGetValues('validate')

            acos = a * np.cos( (self.k @ dx) - self.w @ t )
            bsin = b * np.sin( (self.k @ dx) - self.w @ t )   

            self.reconstructedSurfaceValidate = sumMatrix @ (acos + bsin)

        if intent == 'predict':
            t = np.expand_dims(dm.write.time, axis = 0)
            dx = self.xpred * np.ones((1, dm.write.nSamples))

            a, b = dm.inversionGetValues('predict')

            acos = a * np.cos( (self.k @ dx) - self.w @ t )
            bsin = b * np.sin( (self.k @ dx) - self.w @ t )   

            self.reconstructedSurfacePredict = sumMatrix @ (acos + bsin)
     


    def setVis(self, dm):
        # plt.ion()
        figure, ax = plt.subplots(figsize = (8,5))
        plt.ylim([-.2, .2])
        ax.axvline(0, color = 'gray', linestyle = '-', label = 'reconstruction time')
        reconstructedLine, = ax.plot(dm.validateWrite.time, np.zeros(dm.validateWrite.nSamples), color = 'blue', label = 'reconstructed')
        measuredLine, = ax.plot(dm.validateRead.time, np.zeros(dm.validateRead.nSamples), color = 'red', label = 'measured')
        
        tMin = ax.axvline(-1, color = 'black', linestyle = '--', label = 'reconstruction boundary')
        tMax = ax.axvline(1, color = 'black', linestyle = '--')
        
        plt.title("Reconstruction and propagation loaded incrementally")
        plt.xlabel("time (s)")
        plt.ylabel("height (m)")
        ax.legend(loc = 'upper left')
        V = figure, ax, reconstructedLine, measuredLine, tMin, tMax

        return V

    def updateVis(self, dm, V):

        figure, ax, reconstructedLine, measuredLine, tMin, tMax = V

        try:
            reconstructedLine.set_ydata(np.squeeze(self.reconstructedSurfaceValidate))
        except AttributeError:
            reconstructedLine.set_ydata(np.zeros(dm.validateWrite.nSamples))
        
        measuredLine.set_ydata(np.squeeze(dm.validateData())) # this is at the readDT
        
        try:
            tMin.set_xdata(self.t_min)
            tMax.set_xdata(self.t_max)
        except AttributeError:
            tMin.set_xdata(-dm.updateInterval)
            tMax.set_xdata(dm.updateInterval)

        figure.canvas.draw()
        figure.canvas.flush_events()

        self.plotFlag = False


    def filter(self):
        # do some lowpass filtering on noisy data
        pass
    def update_measurement_locations(self):
        # hold the locations x in the wrp class and update them if necessary
        pass


class DataLoader:
    """Hands data from a static file to DataManager
    
    Args:
        dataFile: location of csv file with wave measurements, samples 
                    are columns and locations are rows
        timeFile: csv of time stamps associated with dataFile
    """
    def __init__(self, dataFile, timeFile):
        # location of data
        self.dataFileName = dataFile
        # load full array
        self.dataFull = genfromtxt(self.dataFileName, delimiter=',')

        # location of data
        self.timeFileName = timeFile
        # load full array
        self.timeFull = genfromtxt(self.timeFileName, delimiter=',')

        # location in full array for dynamic method
        self.currentIndex = 0

        # location in full array for dynamic soft method
        self.bufferCurrentIndex = 0
        self.validateCurrentIndex = 0

    def generateBuffersStatic(self, dm, reconstructionTime):
        """Goes to specified time in static file and assigns data accordingly
        
        Args: 
            dm: instance of DataManager
            reconstructionTime: time at which to reconstruct
        """
        # load reconstruction and validation data one time

        # index of the specified reconstruction time
        self.reconstructionIndex = np.argmin( np.abs(reconstructionTime - self.timeFull))
        bufferLowIndex = self.reconstructionIndex - dm.buffer.nSamples
        bufferHighIndex = self.reconstructionIndex

        validateLowIndex = self.reconstructionIndex - dm.validateNPastSamples
        validateHighIndex = self.reconstructionIndex + dm.validateNFutureSamples

        dm.buffer.values = self.dataFull[:, bufferLowIndex:bufferHighIndex]
        dm.validateValues = self.dataFull[:, validateLowIndex:validateHighIndex]

        dm.predictValues = self.dataFull[dm.pg, self.reconstructionIndex:self.reconstructionIndex + dm.readRate*dm.updateInterval]

    def generateBuffersDynamic(self, dm, wrp, reconstructionTime, callFunc):
        """Reads data file in chunks
        
        Starts from beginning of file, and iterates through assigning each 
        chunk to the stored buffers as it goes. Calls a function at each
        step.
        
        Args: 
            dm: instance of DataManager
            wrp: instance of WRP
            reconstructionTime: time at which to stop taking data
            callFunc: function to call at every step
        """
        # index of the specified reconstruction time
        self.reconstructionIndex = np.argmin( np.abs(reconstructionTime - self.timeFull))

        while self.currentIndex <= self.reconstructionIndex:
            if self.currentIndex == 0:

                filler = np.reshape(np.cos(dm.buffer.time + np.random.normal()), (1, dm.buffer.nSamples))
                fillAll = np.tile(filler, (4, 1))
                dm.buffer.values = fillAll

                newData = self.dataFull[:, :dm.read.nSamples]

            else:              
                newData = self.dataFull[:, self.currentIndex:self.currentIndex + dm.read.nSamples]
            
            dm.bufferUpdate(newData)
            dm.validateUpdate(newData)

            # decide what to do on update in main script
            callFunc(wrp, dm)
            
            self.newMeasurement = newData[3,:]
            self.currentIndex += dm.read.nSamples

    def generateBuffersDynamicSoft(self, dm, reconstructionTime):
        # called 'soft' because it still allocates data to the right place in each buffer, 
        # unlike true acquisition which needs to do reconstruction as soon as data is available

        # index of the specified reconstruction time
        self.reconstructionIndex = np.argmin( np.abs(reconstructionTime - self.timeFull))

        # add samples from self.dataFull to dm.validateValues until the number of samples added
        # matches dm.validateNFutureSamples
        dm.validateValues[:, -dm.validateNFutureSamples:] = self.dataFull[:, :dm.validateNFutureSamples]
        self.validateCurrentIndex = dm.validateNFutureSamples

        while self.bufferCurrentIndex < self.reconstructionIndex:
            if self.bufferCurrentIndex == 0:
                bufferNewData = self.dataFull[:, :dm.read.nSamples]
                dm.bufferUpdate(bufferNewData)

            # update buffer.values and validateValues
            bufferNewData = self.dataFull[:, self.bufferCurrentIndex: self.bufferCurrentIndex+dm.read.nSamples]
            dm.bufferUpdate(bufferNewData)

            validateNewData = self.dataFull[:, self.validateCurrentIndex: self.validateCurrentIndex+dm.read.nSamples]
            dm.validateUpdate(validateNewData)

            self.bufferCurrentIndex += dm.read.nSamples
            self.validateCurrentIndex += dm.read.nSamples

        print(dm.buffer.values)