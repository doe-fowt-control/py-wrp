class channel:
    def __init__(self, port, calibration_slope):
        self.port = port
        self.calibration_slope = calibration_slope

class WaveGauges(channel):
    def __init__(self, port, calibration_slope, position, role):
        super().__init__(port, calibration_slope)
        self.position = position
        self.role = role

