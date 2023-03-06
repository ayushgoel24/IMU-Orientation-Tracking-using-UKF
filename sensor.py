import numpy as np

class Sensor( object ):

    def __init__( self, sensor_measurement, sensitivities: list, biases: list, multiplying_factor: float ) -> None:
        self.sensor_measurement = sensor_measurement
        self.sensitivities =  np.array( sensitivities ).reshape( -1, 1 )
        self.biases = np.array( biases ).reshape( -1, 1 )
        self.multiplying_factor = multiplying_factor

    def rectify_axes( self, metric_data ):
        pass

    def adc_to_metric( self ):
        assert len( self.sensitivities ) == 3, "Sensitivities must contain 3 values corresponding to X, Y and Z axis"
        assert len( self.biases ) == 3, "Biases must contain 3 values corresponding to X, Y and Z axis"
        assert self.sensor_measurement.shape[0] == 3, "Sensor data must contain 3 rows corresponding to X, Y and Z axis"

        metric_data = np.zeros( self.sensor_measurement.shape )
        scale = ( self.multiplying_factor * 3300.0 ) / 1023.0
        metric_data = ( self.sensor_measurement - self.biases ) * scale / self.sensitivities

        return self.rectify_axes( metric_data )
    
    def caliberate( self ):
        pass