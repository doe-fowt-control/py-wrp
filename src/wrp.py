import numpy as np
from scipy.signal import welch
from scipy import linalg
import matplotlib.pyplot as plt
from scipy import interpolate
from numpy import genfromtxt
import time

class Prams:
    """Parameters which alter the inversion and reconstruction process
    These are specifically WRP parameters
    
    Attributes:
        nf: number of frequencies to use for reconstruction
        mu: threshold parameter to determine fastest/slowest 
            group velocities for prediction zone
        lam: regularization parameter for least squares fit
        ta: reconstruction assimilation time
        ts: spectral assimilation time
        postWindow: validation reconstruction after reconstruction time
        preWindow: validation reconstruction before reconstruction time
        use_full_spectrum: for post-processing, whether to read the full data file to determine spectral characteristics
        resample: whether or not to resample data (default no, should be taken care of with DAQ configuration)
    """
    def __init__(
        self,
        mg = [1,2,3,4,5,6,7],
        pg = 8,
        nf = 100,
        mu = 0.05,
        lam = 1,
        ta = 15,
        ts = 30,
        postWindow = 5,
        preWindow = 5,
        use_full_spectrum = 1,
        resample = 0,
        updateInterval = 1,
        wrpRate = 100,
        controlRate = 20,
        hp = 6,
        mpc_ts = 0.02,
    ):
        self.mg = mg
        self.pg = pg
        self.nf = nf        # number of harmonics in reconstruction
        self.mu = mu        # cutoff threshold for prediction zone
        self.lam = lam      # regularization in inverse problem
        self.ta = ta 
        self.ts = ts
        self.updateInterval = updateInterval
        
        self.wrpRate = wrpRate
        self.wrpDT = (1/self.wrpRate)

        self.controlRate = controlRate
        self.controlDT = (1/self.controlRate)

        self.hp = hp
        self.mpc_ts = mpc_ts

        # # round up window to nearest multiple
        # if postWindow % updateInterval != 0: 
        #     postWindow = postWindow + (updateInterval - (postWindow % updateInterval))
        
        self.postWindow = postWindow
        self.preWindow = preWindow

        self.use_full_spectrum = use_full_spectrum
        self.resample = resample


class Controller:
    def __init__(
            self,
            hp,
            prediction_horizon_times,
            mv,
            u_ekf,
            predicted_amplitude,
            predicted_ang_freq,
            predicted_phase,
            predicted_wave_elevation,
            # predicted_amplitude_new,
            # predicted_ang_freq_new,
            # predicted_phase_new,
            # predicted_wave_elevation_new,            
            float_excitation_data, 
            parameters_cell_array, 
            onlineData, 
            ekf,
            y,
            x, 
            controller_rack_length, 
            n_x, 
            n_y, 
            n_md, 
            nlobj,
    ):
        self.hp = hp
        self.mv = mv
        self.prediction_horizon_times = prediction_horizon_times
        self.u_ekf = u_ekf
        self.predicted_amplitude = predicted_amplitude,
        self.predicted_ang_freq = predicted_ang_freq,
        self.predicted_phase = predicted_phase,
        self.predicted_wave_elevation = predicted_wave_elevation,

        # self.predicted_amplitude_new = predicted_amplitude_new,
        # self.predicted_ang_freq_new = predicted_ang_freq_new,
        # self.predicted_phase_new = predicted_phase_new,
        # self.predicted_wave_elevation_new = predicted_wave_elevation_new,

        self.float_excitation_data = float_excitation_data
        self.parameters_cell_array = parameters_cell_array
        self.onlineData = onlineData
        self.ekf = ekf
        self.y = y
        self.x = x
        self.controller_rack_length = controller_rack_length
        self.n_x = n_x
        self.n_y = n_y
        self.n_md = n_md
        self.nlobj = nlobj


# this was and still should be a part of wrp
    #     self.new_ready = 0

    # def cycleControlArrays(self):
    #     # out with the old, in with the new
    #     self.predicted_amplitude_old = self.predicted_amplitude_new,
    #     self.predicted_ang_freq_old = self.predicted_ang_freq_new,
    #     self.predicted_phase_old = self.predicted_phase_new,
    #     self.predicted_wave_elevation_old = self.predicted_wave_elevation_new,
        

    #     # out with the new, in with the fresh
    #     self.controlElevationTimeSeriesNew = self.reconstructedSurfacePredict
    #     self.controlAmplitudesNew = self.A
    #     self.controlFrequenciesNew = self.w
    #     self.controlPhasesNew = self.phi

    #     time.sleep(0.5)

    #     # declare new data available
    #     self.new_ready = 1

class Sensors:
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

    def addSensor(self, position, slope = 1, name = "-", role = 0):
        """Adds details for an add gauge to class

        Args:
            position: physical location in space, meters
            slope: conversion factor for measurement m/V
            name: analog channel/ address, string
            role: 0 for measurement, 1 for validation

        """
        self.xPositions.append(round(position, 4))
        self.calibrationSlopes.append(slope)
        self.portNames.append(name)
        self.wrpRole.append(role)

    def nSensors(self):
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
        pg = [i for i, e in enumerate(self.wrpRole) if e == 1]
        return pg
    
    def stringPotIndex(self):
        """Find indices of string potentiometers for roll and heave measurements
        
        Returns:
            A list of indices
        """
        sp = [i for i, e in enumerate(self.wrpRole) if e == 2]
        return sp





class TaskManager:
    """Static arrays and their corresponding time series"""
    def __init__(
        self,
        sensors,
        sampleRate = 100,
        handoffInterval = 1,
        bufferStartTime = -1,
        bufferEndTime = 1,
    ):

        # unpack timing attributes
        self.sampleRate = sampleRate
        self.handoffInterval = handoffInterval
        self.bufferStartTime = bufferStartTime
        self.bufferEndTime = bufferEndTime

        # steal sensors attributes
        self.xPositions = sensors.xPositions
        self.calibrationSlopes = sensors.calibrationSlopes
        self.portNames = sensors.portNames
        self.wrpRole = sensors.wrpRole
        self.mg = sensors.measurementIndex()
        self.pg = sensors.predictionIndex()

        # define alternative important timing attributes
        self.dt = 1/sampleRate
        self.duration = bufferEndTime - bufferStartTime

        # counts sensors which were given to TaskManager as argument
        self.nRows = sensors.nSensors()

        # set up read - samples, values, time
        self.nSamples = int(sampleRate * self.duration) # int to account for non integer update intervals
        self.nHandoffSamples = int(sampleRate * handoffInterval)

        # full buffer stored locally on PC
        self.time = np.arange(bufferStartTime, bufferEndTime, self.dt)
        self.bufferValues = np.zeros((self.nRows, self.nSamples), dtype=np.float64)
        # array for hardware buffer to deliver values
        self.handoffValues = np.zeros((self.nRows, int(handoffInterval * sampleRate)), dtype=np.float64)
        

        [self.Xmesh, self.Tmesh] = np.meshgrid(self.xPositions, self.time, indexing='xy')


    def bufferUpdate(self):
        """adds data present in handoffValues to the end of bufferValues
        by shifting existing data and removing the oldest
        

        """
        # shift old data to the end of the matrix
        self.bufferValues = np.roll(self.bufferValues, -self.nHandoffSamples)
        # write over old data with new data
        self.bufferValues[:, -self.nHandoffSamples:] = self.handoffValues

    def preprocess(self, data, whichSensors, newRate = "default"):
        """scales data by calibration constants and subtracts the mean
        
        Args: 
            data: array of values to be processed
            whichSensors: the indices of the gauges which correspond to the data being processed
        
        Returns: 
            array of processed data
        """

        # downsample to specified rate if given
        if newRate != "default":
            # check that new rate is actually lower than the given sample rate
            if newRate < self.sampleRate:
                resample_interval = int(self.sampleRate / newRate)
                data = data[:, 0::resample_interval]
            else:
                print("WRP rate is less than sample rate")

        # scale by calibration constants
        scale = np.expand_dims(np.array(self.calibrationSlopes)[whichSensors], axis = 1)
        data *= scale

        # center on mean
        dataMeans = np.expand_dims(np.mean(data, axis = 1), axis = 1)
        data -= dataMeans

        return data
    
    def spectralData(self):
        """gets data from buffer.values which should be used as spectral data
        
        Calls preprocessing to scale and center data on mean

        Returns:
            An array of processed data for spectral information
        """
        # get relevant data
        data = self.bufferValues[self.mg, :]

        # center on mean
        processedData = self.preprocess(data, self.mg)

        return processedData
    
    def reconstructionData(self, ta):
        """gets data from bufferValues which should be used for reconstruction
        
        Calls preprocessing to scale and center data on mean

        Args:
            ta: assimilation time
        
        Returns:
            An array of processed data for reconstruction
        """
        # determine number of past samples to use from `ta`
        assimilationSamples = ta * self.sampleRate

        # select measurement gauges across reconstruction time
        data = self.bufferValues[self.mg, -assimilationSamples:]

        processedData = self.preprocess(data, self.mg)
        return processedData




class WRP:
    """Implements methods of wave reconstruction and propagation

    Args: 
        gauges: instance of Sensors
    """
    def __init__(self, pram, wtm):
# unpack parameter object
        self.mu = pram.mu
        self.lam = pram.lam
        self.nf = pram.nf
        self.ta = pram.ta
        self.ts = pram.ts
        self.postWindow = pram.postWindow
        self.preWindow = pram.preWindow
        self.updateInterval = pram.updateInterval
        
        self.wrpRate = pram.wrpRate
        self.wrpDT = pram.wrpDT

        self.controlRate = pram.controlRate
        self.controlDT = pram.controlDT

        self.hp = pram.hp
        self.mpc_ts = pram.mpc_ts

# unpack waveTaskManager object
        self.x = wtm.xPositions
        self.calibration = wtm.calibrationSlopes
        self.mg = wtm.mg    # gauges to select for reconstruction
        self.pg = wtm.pg    # gauges for prediction
        # self.updateInterval = wtm.handoffInterval

        self.inversionNSaved = int(self.postWindow / self.updateInterval) + 1
        self.inversionSavedValues = np.zeros((2, self.inversionNSaved, pram.nf))

        # important spatial parameters for wrp based on gauge locations
        self.xmax = np.max( np.array(self.x)[self.mg] )
        self.xmin = np.min( np.array(self.x)[self.mg] )
        self.xpred = np.array(self.x)[self.pg]

        self.plotFlag = False

        # flags for trigger from read_callback
        self.controlIteration = 0
        self.controlIterationLimit = int(self.updateInterval / self.controlDT)

        # flag indicating if a new prediction is ready
        self.new_ready = 0

        # elevation time series is twice the length of the update interval
        self.controlTime = np.arange(0, 2*self.updateInterval, self.controlDT)

        self.controlElevationTimeSeriesOld = np.zeros((self.hp, 1))
        self.controlAmplitudesOld = np.zeros((1, self.nf))
        self.controlFrequenciesOld = np.zeros((1, self.nf))
        self.controlPhasesOld = np.zeros((1, self.nf))
        self.eta_of_t_old = lambda x : x

        self.controlElevationTimeSeriesNew = np.zeros((self.hp, 1))
        self.controlAmplitudesNew = np.zeros((1, self.nf))
        self.controlFrequenciesNew = np.zeros((1, self.nf))
        self.controlPhasesNew = np.zeros((1, self.nf))
        self.eta_of_t_new = lambda x : x
    
    def cycleControlArrays(self):
        # out with the old, in with the new
        self.controlElevationTimeSeriesOld = self.controlElevationTimeSeriesNew
        self.controlAmplitudesOld = self.controlAmplitudesNew
        self.controlFrequenciesOld = self.controlFrequenciesNew
        self.controlPhasesOld = self.controlPhasesNew
        self.eta_of_t_old = self.eta_of_t_new

    def updateControlArrays(self):
        # out with the new, in with the fresh
        self.controlElevationTimeSeriesNew = self.reconstructedSurfacePredict
        self.controlAmplitudesNew = self.A
        self.controlFrequenciesNew = self.w
        self.controlPhasesNew = self.phi
        self.eta_of_t_new = interpolate.interp1d(self.controlTime, self.reconstructedSurfacePredict)

        # time.sleep(0.5)

        # # declare new data available (move this to main script)
        # self.new_ready = 1





    def spectral(self, wtm):
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
        # print('spectral data requested')

        # assign spectral variables to wrp class
        data = wtm.spectralData()

        # print('spectral data acquired')
        # print(np.shape(data))

        # check to see if the buffer is filled
        self.bufferFilled = data[0][0] != data[0][1]
        self.bufferFilled = 1
        # print(self.bufferFilled)


        if self.bufferFilled:
            f, pxxEach = welch(data, fs = wtm.sampleRate)
            pxx = np.mean(pxxEach, 0)
            self.w = f * np.pi * 2

            self.T_p = 1 / (f[pxx == np.max(pxx)])

            # peak wavelength
            self.k_p = (1 / 9.81) * (2 * np.pi / self.T_p)**2

            # zero-th moment as area under spectral curve
            self.m0 = np.trapz(pxx, f)


            # significant wave height from zero moment
            self.Hs = 4 * np.sqrt(self.m0)

            # identify region meeting energy threshold
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


    def lwt(self, wtm):
        """Runs complete set of in-the-loop operations for linear wave theory
        
        # this single call to the icwm method does a bunch of stuff
        # - evaluation of the spectrum in its current state
        # - inversion at the current time step, save inversion values
        # - reconstruct at the time interval needed for the control system
        # - reconstruct an old time series for the validation plot
        """
        self.spectral(wtm)

        print('spectrum calculated')
        
        # only do the actions once enough data is available to evaluate the spectrum
        if self.bufferFilled:
            self.inversion_lwt(wtm)
            self.reconstruct_lwt(wtm, 'validate')
            self.reconstruct_lwt(wtm, 'predict')


    def inversion_lwt(self, wtm):
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
        eta = wtm.reconstructionData(self.ta)
        # print(np.shape(eta))
        t = np.arange(-self.ta, 0, self.wrpDT)
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

        self.inversionUpdate(a, b)


    def reconstruct_lwt(self, wtm, intent):
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
            validateTime = np.arange(-self.ta - self.preWindow, self.postWindow, self.wrpDT)
            t = np.expand_dims(validateTime, axis = 0)
            # print(self.xpred)
            dx = self.xpred * np.ones((1, len(validateTime)))
            
            a, b = self.inversionGetValues('validate')

            acos = a * np.cos( (self.k @ dx) - self.w @ t )
            bsin = b * np.sin( (self.k @ dx) - self.w @ t )   

            self.reconstructedSurfaceValidate = sumMatrix @ (acos + bsin)

        if intent == 'predict':
            # predictTime = np.arange(0, self.postWindow, self.wrpDT)
            # t = np.expand_dims(predictTime, axis = 0)
            # t = np.expand_dims(mpctm.time, axis = 0)
            t = np.expand_dims(self.controlTime, axis = 0)
            dx = self.xpred * np.ones((1, len(t)))

            a, b = self.inversionGetValues('predict')
            
            # print(np.shape(t))
            # print(np.shape(dx))
            # print(np.shape(a))
            # print(np.shape(b))

            acos = a * np.cos( (self.k @ dx) - self.w @ t )
            bsin = b * np.sin( (self.k @ dx) - self.w @ t )   

            self.reconstructedSurfacePredict = sumMatrix @ (acos + bsin)
     

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
        
    # def setVis(self, dm):
    #     # plt.ion()
    #     figure, ax = plt.subplots(figsize = (8,5))
    #     plt.ylim([-.2, .2])
    #     ax.axvline(0, color = 'gray', linestyle = '-', label = 'reconstruction time')
    #     reconstructedLine, = ax.plot(dm.validateWrite.time, np.zeros(dm.validateWrite.nSamples), color = 'blue', label = 'reconstructed')
    #     measuredLine, = ax.plot(dm.validateRead.time, np.zeros(dm.validateRead.nSamples), color = 'red', label = 'measured')
        
    #     tMin = ax.axvline(-1, color = 'black', linestyle = '--', label = 'reconstruction boundary')
    #     tMax = ax.axvline(1, color = 'black', linestyle = '--')
        
    #     plt.title("Reconstruction and propagation loaded incrementally")
    #     plt.xlabel("time (s)")
    #     plt.ylabel("height (m)")
    #     ax.legend(loc = 'upper left')
    #     V = figure, ax, reconstructedLine, measuredLine, tMin, tMax

    #     return V

    # def updateVis(self, dm, V):

    #     figure, ax, reconstructedLine, measuredLine, tMin, tMax = V

    #     try:
    #         reconstructedLine.set_ydata(np.squeeze(self.reconstructedSurfaceValidate))
    #     except AttributeError:
    #         reconstructedLine.set_ydata(np.zeros(dm.validateWrite.nSamples))
        
    #     measuredLine.set_ydata(np.squeeze(dm.validateData())) # this is at the readDT
        
    #     try:
    #         tMin.set_xdata(self.t_min)
    #         tMax.set_xdata(self.t_max)
    #     except AttributeError:
    #         tMin.set_xdata(-dm.updateInterval)
    #         tMax.set_xdata(dm.updateInterval)

    #     figure.canvas.draw()
    #     figure.canvas.flush_events()

    #     self.plotFlag = False


    # def filter(self):
    #     # do some lowpass filtering on noisy data
    #     pass
    # def update_measurement_locations(self):
    #     # hold the locations x in the wrp class and update them if necessary
    #     pass


class DataLoader:
    """Hands data from a Static file to DataManager
    
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
        """Goes to specified time in Static file and assigns data accordingly
        
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


# import numpy as np
# from scipy.signal import welch
# from scipy import linalg
# import matplotlib.pyplot as plt
# from numpy import genfromtxt
# import time

# class Prams:
#     """Parameters which alter the inversion and reconstruction process
#     These are specifically WRP parameters
    
#     Attributes:
#         nf: number of frequencies to use for reconstruction
#         mu: threshold parameter to determine fastest/slowest 
#             group velocities for prediction zone
#         lam: regularization parameter for least squares fit
#         ta: reconstruction assimilation time
#         ts: spectral assimilation time
#         postWindow: validation reconstruction after reconstruction time
#         preWindow: validation reconstruction before reconstruction time
#         use_full_spectrum: for post-processing, whether to read the full data file to determine spectral characteristics
#         resample: whether or not to resample data (default no, should be taken care of with DAQ configuration)
#     """
#     def __init__(
#         self,
#         mg = [1,2,3,4,5,6,7],
#         pg = 8,
#         nf = 100,
#         mu = 0.05,
#         lam = 1,
#         ta = 15,
#         ts = 30,
#         postWindow = 5,
#         preWindow = 5,
#         use_full_spectrum = 1,
#         resample = 0,
#         updateInterval = 1,
#         wrpRate = 100,
#         controlRate = 20,
#     ):
#         self.mg = mg
#         self.pg = pg
#         self.nf = nf        # number of harmonics in reconstruction
#         self.mu = mu        # cutoff threshold for prediction zone
#         self.lam = lam      # regularization in inverse problem
#         self.ta = ta 
#         self.ts = ts
#         self.updateInterval = updateInterval
        
#         self.wrpRate = wrpRate
#         self.wrpDT = (1/self.wrpRate)

#         self.controlRate = controlRate
#         self.controlDT = (1/self.controlRate)

#         # # round up window to nearest multiple
#         # if postWindow % updateInterval != 0: 
#         #     postWindow = postWindow + (updateInterval - (postWindow % updateInterval))
        
#         self.postWindow = postWindow
#         self.preWindow = preWindow

#         self.use_full_spectrum = use_full_spectrum
#         self.resample = resample


# class Controller:
#     def __init__(
#             self,
#             hp,
#             prediction_horizon_times,
#             mv,
#             u_ekf,
#             predicted_amplitude_old,
#             predicted_ang_freq_old,
#             predicted_phase_old,
#             predicted_wave_elevation_old,
#             predicted_amplitude_new,
#             predicted_ang_freq_new,
#             predicted_phase_new,
#             predicted_wave_elevation_new,            
#             float_excitation_data, 
#             parameters_cell_array, 
#             onlineData, 
#             ekf,
#             y,
#             x, 
#             controller_rack_length, 
#             n_x, 
#             n_y, 
#             n_md, 
#             nlobj,
#     ):
#         self.hp = hp
#         self.mv = mv
#         self.prediction_horizon_times = prediction_horizon_times
#         self.u_ekf = u_ekf
#         self.predicted_amplitude_old = predicted_amplitude_old,
#         self.predicted_ang_freq_old = predicted_ang_freq_old,
#         self.predicted_phase_old = predicted_phase_old,
#         self.predicted_wave_elevation_old = predicted_wave_elevation_old,
#         self.predicted_amplitude_new = predicted_amplitude_new,
#         self.predicted_ang_freq_new = predicted_ang_freq_new,
#         self.predicted_phase_new = predicted_phase_new,
#         self.predicted_wave_elevation_new = predicted_wave_elevation_new,
#         self.float_excitation_data = float_excitation_data
#         self.parameters_cell_array = parameters_cell_array
#         self.onlineData = onlineData
#         self.ekf = ekf
#         self.y = y
#         self.x = x
#         self.controller_rack_length = controller_rack_length
#         self.n_x = n_x
#         self.n_y = n_y
#         self.n_md = n_md
#         self.nlobj = nlobj

# class Sensors:
#     """The information needed to a interpret a physical wave measurement
    
#     Contains lists with the physical locations, calibration constants, analog
#     port, and role in the WRP algorithm, which can be either `measurement` or 
#     `validation`. The latter also indicates the physical location of the float.
#     Position is defined where waves move in positive x direction and x = 0 at the
#     validation gauge.

#     Attributes:
#         xPositions: list of physical locations
#         calibrationSlopes: list of conversion factors from measurement to wave height, (m/V)
#         portNames: list of channels on which to aquire data from this wave gauge
#         wrpRole: list of roles, 0 indicates measurement, 1 indicates validation
#     """
#     def __init__(self):
#         """Set up lists for pertinent wave gauge information
#         """
#         self.xPositions = []
#         self.calibrationSlopes = []
#         self.portNames = []
#         self.wrpRole = []      # 0 = measurement gauge, 1 = prediction gauge

#     def addSensor(self, position, slope = 1, name = "-", role = 0):
#         """Adds details for an add gauge to class

#         Args:
#             position: physical location in space, meters
#             slope: conversion factor for measurement m/V
#             name: analog channel/ address, string
#             role: 0 for measurement, 1 for validation

#         """
#         self.xPositions.append(round(position, 4))
#         self.calibrationSlopes.append(slope)
#         self.portNames.append(name)
#         self.wrpRole.append(role)

#     def nSensors(self):
#         """Determine number of gauges which have been added"""
#         return(len(self.xPositions))

#     def measurementIndex(self):
#         """Find indices of gauges used for measurement
        
#         Returns:
#             A list of indices
#         """
#         mg = [i for i, e in enumerate(self.wrpRole) if e == 0]
#         return mg

#     def predictionIndex(self):
#         """Find indices of gauges used for validation
        
#         Returns:
#             A list of indices
#         """
#         pg = [i for i, e in enumerate(self.wrpRole) if e == 1]
#         return pg
    
#     def stringPotIndex(self):
#         """Find indices of string potentiometers for roll and heave measurements
        
#         Returns:
#             A list of indices
#         """
#         sp = [i for i, e in enumerate(self.wrpRole) if e == 2]
#         return sp





# class TaskManager:
#     """Static arrays and their corresponding time series"""
#     def __init__(
#         self,
#         sensors,
#         sampleRate = 100,
#         handoffInterval = 1,
#         bufferStartTime = -1,
#         bufferEndTime = 1,
#     ):

#         # unpack timing attributes
#         self.sampleRate = sampleRate
#         self.handoffInterval = handoffInterval
#         self.bufferStartTime = bufferStartTime
#         self.bufferEndTime = bufferEndTime

#         # steal sensors attributes
#         self.xPositions = sensors.xPositions
#         self.calibrationSlopes = sensors.calibrationSlopes
#         self.portNames = sensors.portNames
#         self.wrpRole = sensors.wrpRole
#         self.mg = sensors.measurementIndex()
#         self.pg = sensors.predictionIndex()

#         # define alternative important timing attributes
#         self.dt = 1/sampleRate
#         self.duration = bufferEndTime - bufferStartTime

#         # counts sensors which were given to TaskManager as argument
#         self.nRows = sensors.nSensors()

#         # set up read - samples, values, time
#         self.nSamples = int(sampleRate * self.duration) # int to account for non integer update intervals
#         self.nHandoffSamples = int(sampleRate * handoffInterval)

#         # full buffer stored locally on PC
#         self.time = np.arange(bufferStartTime, bufferEndTime, self.dt)
#         self.bufferValues = np.zeros((self.nRows, self.nSamples), dtype=np.float64)
#         # array for hardware buffer to deliver values
#         self.handoffValues = np.zeros((self.nRows, int(handoffInterval * sampleRate)), dtype=np.float64)
        

#         [self.Xmesh, self.Tmesh] = np.meshgrid(self.xPositions, self.time, indexing='xy')


#     def bufferUpdate(self):
#         """adds data present in handoffValues to the end of bufferValues
#         by shifting existing data and removing the oldest
        

#         """
#         # shift old data to the end of the matrix
#         self.bufferValues = np.roll(self.bufferValues, -self.nHandoffSamples)
#         # write over old data with new data
#         self.bufferValues[:, -self.nHandoffSamples:] = self.handoffValues

#     def preprocess(self, data, whichSensors, newRate = "default"):
#         """scales data by calibration constants and subtracts the mean
        
#         Args: 
#             data: array of values to be processed
#             whichSensors: the indices of the gauges which correspond to the data being processed
        
#         Returns: 
#             array of processed data
#         """

#         # downsample to specified rate if given
#         if newRate != "default":
#             # check that new rate is actually lower than the given sample rate
#             if newRate < self.sampleRate:
#                 resample_interval = int(self.sampleRate / newRate)
#                 data = data[:, 0::resample_interval]
#             else:
#                 print("WRP rate is less than sample rate")

#         # scale by calibration constants
#         scale = np.expand_dims(np.array(self.calibrationSlopes)[whichSensors], axis = 1)
#         data *= scale

#         # center on mean
#         dataMeans = np.expand_dims(np.mean(data, axis = 1), axis = 1)
#         data -= dataMeans

#         return data
    
#     def spectralData(self):
#         """gets data from buffer.values which should be used as spectral data
        
#         Calls preprocessing to scale and center data on mean

#         Returns:
#             An array of processed data for spectral information
#         """
#         # get relevant data
#         data = self.bufferValues[self.mg, :]

#         # center on mean
#         processedData = self.preprocess(data, self.mg)

#         return processedData
    
#     def reconstructionData(self, ta):
#         """gets data from bufferValues which should be used for reconstruction
        
#         Calls preprocessing to scale and center data on mean

#         Args:
#             ta: assimilation time
        
#         Returns:
#             An array of processed data for reconstruction
#         """
#         # determine number of past samples to use from `ta`
#         assimilationSamples = ta * self.sampleRate

#         # select measurement gauges across reconstruction time
#         data = self.bufferValues[self.mg, -assimilationSamples:]

#         processedData = self.preprocess(data, self.mg)
#         return processedData




# class WRP:
#     """Implements methods of wave reconstruction and propagation

#     Args: 
#         gauges: instance of Sensors
#     """
#     def __init__(self, pram, wtm):
# # unpack parameter object
#         self.mu = pram.mu
#         self.lam = pram.lam
#         self.nf = pram.nf
#         self.ta = pram.ta
#         self.ts = pram.ts
#         self.postWindow = pram.postWindow
#         self.preWindow = pram.preWindow
#         self.updateInterval = pram.updateInterval
        
#         self.wrpRate = pram.wrpRate
#         self.wrpDT = pram.wrpDT

#         self.controlRate = pram.controlRate
#         self.controlDT = pram.controlDT

# # unpack waveTaskManager object
#         self.x = wtm.xPositions
#         self.calibration = wtm.calibrationSlopes
#         self.mg = wtm.mg    # gauges to select for reconstruction
#         self.pg = wtm.pg    # gauges for prediction
#         # self.updateInterval = wtm.handoffInterval

#         self.inversionNSaved = int(self.postWindow / self.updateInterval) + 1
#         self.inversionSavedValues = np.zeros((2, self.inversionNSaved, pram.nf))

#         # important spatial parameters for wrp based on gauge locations
#         self.xmax = np.max( np.array(self.x)[self.mg] )
#         self.xmin = np.min( np.array(self.x)[self.mg] )
#         self.xpred = np.array(self.x)[self.pg]

#         self.plotFlag = False

#         # flags for trigger from read_callback
#         self.controlIteration = 0
#         self.controlIterationLimit = int(self.updateInterval / self.controlDT)

#         # flag indicating if a new prediction is ready
#         self.new_ready = 0

#         # elevation time series is twice the length of the update interval
#         self.controlTime = np.arange(0, 2*self.updateInterval, self.controlDT)

#         self.controlElevationTimeSeriesOld = np.zeros(len(self.controlTime))
#         self.controlAmplitudesOld = np.zeros(self.nf)
#         self.controlFrequenciesOld = np.zeros(self.nf)
#         self.controlPhasesOld = np.zeros(self.nf)

#         self.controlElevationTimeSeriesNew = np.zeros(len(self.controlTime))
#         self.controlAmplitudesNew = np.zeros(self.nf)
#         self.controlFrequenciesNew = np.zeros(self.nf)
#         self.controlPhasesNew = np.zeros(self.nf)
    
#     def cycleControlArrays(self):
#         # out with the old, in with the new
#         self.controlElevationTimeSeriesOld = self.controlElevationTimeSeriesNew
#         self.controlAmplitudesOld = self.controlAmplitudesNew
#         self.controlFrequenciesOld = self.controlFrequenciesNew
#         self.controlPhasesOld = self.controlPhasesNew

#         # out with the new, in with the fresh
#         self.controlElevationTimeSeriesNew = self.reconstructedSurfacePredict
#         self.controlAmplitudesNew = self.A
#         self.controlFrequenciesNew = self.w
#         self.controlPhasesNew = self.phi

#         time.sleep(0.5)

#         # declare new data available
#         self.new_ready = 1





#     def spectral(self, wtm):
#         """Calculates spectral information

#         Uses spectral data to create a set of spectral attributes including \n
#             - T_p: peak period
#             - k_p: peak wavenumber
#             - m0: zero moment of the spectrum
#             - Hs: significant wave height
#             - cg_fast, cg_slow: fastest and slowest group velocity
#             - xe, xb: spatial reconstruction parameters
#             - k_min, k_max: wavenumber bandwidth for reconstruction

#         Args:
#             dm: instance of DataManager
#         """
#         # print('spectral data requested')

#         # assign spectral variables to wrp class
#         data = wtm.spectralData()

#         # print('spectral data acquired')
#         # print(np.shape(data))

#         # check to see if the buffer is filled
#         self.bufferFilled = data[0][0] != data[0][1]
#         self.bufferFilled = 1
#         # print(self.bufferFilled)


#         if self.bufferFilled:
#             f, pxxEach = welch(data, fs = wtm.sampleRate)
#             pxx = np.mean(pxxEach, 0)
#             self.w = f * np.pi * 2

#             self.T_p = 1 / (f[pxx == np.max(pxx)])

#             # peak wavelength
#             self.k_p = (1 / 9.81) * (2 * np.pi / self.T_p)**2

#             # zero-th moment as area under spectral curve
#             self.m0 = np.trapz(pxx, f)


#             # significant wave height from zero moment
#             self.Hs = 4 * np.sqrt(self.m0)

#             # identify region meeting energy threshold
#             thresh = self.mu * np.max(pxx)

#             # set anything above the threshold to zero
#             pxx[pxx > thresh] = 0

#             # print(np.shape(pxx))
#             # plt.plot(f, pxx)
#             # find the locations which didn't make the cut
#             pxxIndex = np.nonzero(pxx)[0]

#             # find the largest gap between nonzero values
#             low_index = np.argwhere( (np.diff(pxxIndex) == np.max(np.diff(pxxIndex))) )[0][0]
#             high_index = pxxIndex[low_index + 1]

#             # plt.axvline(x = f[low_index])
#             # plt.axvline(x = f[high_index])
#             # plt.show()

#             # select group velocities
#             if self.w[low_index] == 0:
#                 self.cg_fast = 20 # super arbitrary
#             else:
#                 self.cg_fast = (9.81 / (self.w[low_index] * 2))
#             self.cg_slow = (9.81 / (self.w[high_index] * 2))

#             # spatial parameters for reconstruction bandwidth
#             self.xe = self.xmax + self.ta * self.cg_slow
#             self.xb = self.xmin

#             # reconstruction bandwidth wavenumbers
#             self.k_min = 2 * np.pi / (self.xe - self.xb)
#             self.k_max = 2 * np.pi / min(abs(np.diff(self.x)))


#     def lwt(self, wtm):
#         """Runs complete set of in-the-loop operations for linear wave theory
        
#         # this single call to the icwm method does a bunch of stuff
#         # - evaluation of the spectrum in its current state
#         # - inversion at the current time step, save inversion values
#         # - reconstruct at the time interval needed for the control system
#         # - reconstruct an old time series for the validation plot
#         """
#         self.spectral(wtm)

#         print('spectrum calculated')
        
#         # only do the actions once enough data is available to evaluate the spectrum
#         if self.bufferFilled:
#             self.inversion_lwt(wtm)
#             self.reconstruct_lwt(wtm, 'validate')
#             self.reconstruct_lwt(wtm, 'predict')


#     def inversion_lwt(self, wtm):
#         """Find linear weights for surface representation

#         Calculates an array of wavenumbers and corresponding deep water
#         frequencies. Solves least squares optimization to get best fit
#         surface representation. Adds results of inversion to a saved 
#         array in DataManager called inversionSavedValues.

#         Args:
#             dm: instance of DataManager
#         """
#     # define wavenumber and frequency range
#         k = np.linspace(self.k_min, self.k_max, self.nf)
#         w = np.sqrt(9.81 * k)


#     # get data
#         eta = wtm.reconstructionData(self.ta)
#         # print(np.shape(eta))
#         t = np.arange(-self.ta, 0, self.wrpDT)
#         x = np.array(self.x)[self.mg]

#     # grid data and reshape for matrix operations
#         X, T = np.meshgrid(x, t)

#         self.k = np.reshape(k, (self.nf, 1))
#         self.w = np.reshape(w, (self.nf, 1))

#         X = np.reshape(X, (1, np.size(X)), order='F')

#         T = np.reshape(T, (1, np.size(T)), order='F')        
#         eta = np.reshape(eta, (np.size(eta), 1))


#         psi = np.transpose(self.k@X - self.w@T)

        
#     # data matrix
#         Z = np.zeros((np.size(X), 2*self.nf))
#         Z[:, :self.nf] = np.cos(psi)
#         Z[:, self.nf:] = np.sin(psi)


#         m = np.transpose(Z)@Z + (np.identity(self.nf * 2) * self.lam)
#         n = np.transpose(Z)@eta
#         weights, res, rnk, s = linalg.lstsq(m, n)

#         # choose all columns [:] for future matrix math
#         a = weights[:self.nf,:]
#         b = weights[self.nf:,:]


#         self.A = np.sqrt(a**2 + b**2)
#         self.phi = np.arctan2(b,a)

#         self.inversionUpdate(a, b)


#     def reconstruct_lwt(self, wtm, intent):
#         """Reconstructs surface using saved inversion values
        
#         Calculates upper and lower limit time boundary for reconstruction time.
#         Calculates shape of reconstructed surface for both validation and 
#         prediction and saves them as attributes of DataManager. The former 
#         is saved as DataManager.reconstructedSurfaceValidate and the latter as
#         DataManager.reconstructedSurfacePredict

#         Args:
#             dm: instance of DataManager
#         """
# # General
#         # prediction zone time boundary
#         self.t_min = (1 / self.cg_slow) * (self.xpred - self.xe)
#         self.t_max = (1 / self.cg_fast) * (self.xpred - self.xb)

#         # matrix for summing across frequencies
#         sumMatrix = np.ones((1, self.nf))

#         if intent == 'validate':
#             validateTime = np.arange(-self.ta - self.preWindow, self.postWindow, self.wrpDT)
#             t = np.expand_dims(validateTime, axis = 0)
#             # print(self.xpred)
#             dx = self.xpred * np.ones((1, len(validateTime)))
            
#             a, b = self.inversionGetValues('validate')

#             acos = a * np.cos( (self.k @ dx) - self.w @ t )
#             bsin = b * np.sin( (self.k @ dx) - self.w @ t )   

#             self.reconstructedSurfaceValidate = sumMatrix @ (acos + bsin)

#         if intent == 'predict':
#             # predictTime = np.arange(0, self.postWindow, self.wrpDT)
#             # t = np.expand_dims(predictTime, axis = 0)
#             # t = np.expand_dims(mpctm.time, axis = 0)
#             t = np.expand_dims(self.controlTime, axis = 0)
#             dx = self.xpred * np.ones((1, len(t)))

#             a, b = self.inversionGetValues('predict')
            
#             # print(np.shape(t))
#             # print(np.shape(dx))
#             # print(np.shape(a))
#             # print(np.shape(b))

#             acos = a * np.cos( (self.k @ dx) - self.w @ t )
#             bsin = b * np.sin( (self.k @ dx) - self.w @ t )   

#             self.reconstructedSurfacePredict = sumMatrix @ (acos + bsin)
     

#     def inversionUpdate(self, a, b):
#         """adds most recent inversion to the end ofinversionSavedValues, deletes the oldest
        
#         Args:
#             a: array of weights for cosine
#             b: array of weights for sine
#         """
#         # array to save backlog of inversion results, good for validating real time
#         self.inversionSavedValues = np.roll(self.inversionSavedValues, -1, axis = 1)

#         # need to squeeze to fit it into the matrix
#         self.inversionSavedValues[0][self.inversionNSaved - 1] = np.squeeze(a)
#         self.inversionSavedValues[1][self.inversionNSaved - 1] = np.squeeze(b)


#     def inversionGetValues(self, method):
#         """retrieves inversion values for a specified `method`
        
#         Args: 
#             method: string
#                 'validate' -> values for validation
#                 'predict' -> most recent values for prediction
        
#         Returns:
#             a: array of weights for cosine
#             b: array of weights for sine
#         """
#         # need expand_dims for the matrix math in reconstruct

#         if method == 'validate':
#             a = np.expand_dims(self.inversionSavedValues[0][0][:], axis=1)
#             b = np.expand_dims(self.inversionSavedValues[1][0][:], axis=1)

#             return a,b

#         if method == 'predict':
#             a = np.expand_dims(self.inversionSavedValues[0][-1][:], axis=1)
#             b = np.expand_dims(self.inversionSavedValues[1][-1][:], axis=1)

#             return a,b
        
#     # def setVis(self, dm):
#     #     # plt.ion()
#     #     figure, ax = plt.subplots(figsize = (8,5))
#     #     plt.ylim([-.2, .2])
#     #     ax.axvline(0, color = 'gray', linestyle = '-', label = 'reconstruction time')
#     #     reconstructedLine, = ax.plot(dm.validateWrite.time, np.zeros(dm.validateWrite.nSamples), color = 'blue', label = 'reconstructed')
#     #     measuredLine, = ax.plot(dm.validateRead.time, np.zeros(dm.validateRead.nSamples), color = 'red', label = 'measured')
        
#     #     tMin = ax.axvline(-1, color = 'black', linestyle = '--', label = 'reconstruction boundary')
#     #     tMax = ax.axvline(1, color = 'black', linestyle = '--')
        
#     #     plt.title("Reconstruction and propagation loaded incrementally")
#     #     plt.xlabel("time (s)")
#     #     plt.ylabel("height (m)")
#     #     ax.legend(loc = 'upper left')
#     #     V = figure, ax, reconstructedLine, measuredLine, tMin, tMax

#     #     return V

#     # def updateVis(self, dm, V):

#     #     figure, ax, reconstructedLine, measuredLine, tMin, tMax = V

#     #     try:
#     #         reconstructedLine.set_ydata(np.squeeze(self.reconstructedSurfaceValidate))
#     #     except AttributeError:
#     #         reconstructedLine.set_ydata(np.zeros(dm.validateWrite.nSamples))
        
#     #     measuredLine.set_ydata(np.squeeze(dm.validateData())) # this is at the readDT
        
#     #     try:
#     #         tMin.set_xdata(self.t_min)
#     #         tMax.set_xdata(self.t_max)
#     #     except AttributeError:
#     #         tMin.set_xdata(-dm.updateInterval)
#     #         tMax.set_xdata(dm.updateInterval)

#     #     figure.canvas.draw()
#     #     figure.canvas.flush_events()

#     #     self.plotFlag = False


#     # def filter(self):
#     #     # do some lowpass filtering on noisy data
#     #     pass
#     # def update_measurement_locations(self):
#     #     # hold the locations x in the wrp class and update them if necessary
#     #     pass


# class DataLoader:
#     """Hands data from a Static file to DataManager
    
#     Args:
#         dataFile: location of csv file with wave measurements, samples 
#                     are columns and locations are rows
#         timeFile: csv of time stamps associated with dataFile
#     """
#     def __init__(self, dataFile, timeFile):
#         # location of data
#         self.dataFileName = dataFile
#         # load full array
#         self.dataFull = genfromtxt(self.dataFileName, delimiter=',')

#         # location of data
#         self.timeFileName = timeFile
#         # load full array
#         self.timeFull = genfromtxt(self.timeFileName, delimiter=',')

#         # location in full array for dynamic method
#         self.currentIndex = 0

#         # location in full array for dynamic soft method
#         self.bufferCurrentIndex = 0
#         self.validateCurrentIndex = 0

#     def generateBuffersStatic(self, dm, reconstructionTime):
#         """Goes to specified time in Static file and assigns data accordingly
        
#         Args: 
#             dm: instance of DataManager
#             reconstructionTime: time at which to reconstruct
#         """
#         # load reconstruction and validation data one time

#         # index of the specified reconstruction time
#         self.reconstructionIndex = np.argmin( np.abs(reconstructionTime - self.timeFull))
#         bufferLowIndex = self.reconstructionIndex - dm.buffer.nSamples
#         bufferHighIndex = self.reconstructionIndex

#         validateLowIndex = self.reconstructionIndex - dm.validateNPastSamples
#         validateHighIndex = self.reconstructionIndex + dm.validateNFutureSamples

#         dm.buffer.values = self.dataFull[:, bufferLowIndex:bufferHighIndex]
#         dm.validateValues = self.dataFull[:, validateLowIndex:validateHighIndex]

#         dm.predictValues = self.dataFull[dm.pg, self.reconstructionIndex:self.reconstructionIndex + dm.readRate*dm.updateInterval]

#     def generateBuffersDynamic(self, dm, wrp, reconstructionTime, callFunc):
#         """Reads data file in chunks
        
#         Starts from beginning of file, and iterates through assigning each 
#         chunk to the stored buffers as it goes. Calls a function at each
#         step.
        
#         Args: 
#             dm: instance of DataManager
#             wrp: instance of WRP
#             reconstructionTime: time at which to stop taking data
#             callFunc: function to call at every step
#         """
#         # index of the specified reconstruction time
#         self.reconstructionIndex = np.argmin( np.abs(reconstructionTime - self.timeFull))

#         while self.currentIndex <= self.reconstructionIndex:
#             if self.currentIndex == 0:

#                 filler = np.reshape(np.cos(dm.buffer.time + np.random.normal()), (1, dm.buffer.nSamples))
#                 fillAll = np.tile(filler, (4, 1))
#                 dm.buffer.values = fillAll

#                 newData = self.dataFull[:, :dm.read.nSamples]

#             else:              
#                 newData = self.dataFull[:, self.currentIndex:self.currentIndex + dm.read.nSamples]
            
#             dm.bufferUpdate(newData)
#             dm.validateUpdate(newData)

#             # decide what to do on update in main script
#             callFunc(wrp, dm)
            
#             self.newMeasurement = newData[3,:]
#             self.currentIndex += dm.read.nSamples

#     def generateBuffersDynamicSoft(self, dm, reconstructionTime):
#         # called 'soft' because it still allocates data to the right place in each buffer, 
#         # unlike true acquisition which needs to do reconstruction as soon as data is available

#         # index of the specified reconstruction time
#         self.reconstructionIndex = np.argmin( np.abs(reconstructionTime - self.timeFull))

#         # add samples from self.dataFull to dm.validateValues until the number of samples added
#         # matches dm.validateNFutureSamples
#         dm.validateValues[:, -dm.validateNFutureSamples:] = self.dataFull[:, :dm.validateNFutureSamples]
#         self.validateCurrentIndex = dm.validateNFutureSamples

#         while self.bufferCurrentIndex < self.reconstructionIndex:
#             if self.bufferCurrentIndex == 0:
#                 bufferNewData = self.dataFull[:, :dm.read.nSamples]
#                 dm.bufferUpdate(bufferNewData)

#             # update buffer.values and validateValues
#             bufferNewData = self.dataFull[:, self.bufferCurrentIndex: self.bufferCurrentIndex+dm.read.nSamples]
#             dm.bufferUpdate(bufferNewData)

#             validateNewData = self.dataFull[:, self.validateCurrentIndex: self.validateCurrentIndex+dm.read.nSamples]
#             dm.validateUpdate(validateNewData)

#             self.bufferCurrentIndex += dm.read.nSamples
#             self.validateCurrentIndex += dm.read.nSamples

#         print(dm.buffer.values)