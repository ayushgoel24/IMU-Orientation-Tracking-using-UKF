import numpy as np
from sensor import Sensor

class Gyroscope( Sensor ):

    def __init__( self, sensor_measurement, sensitivities, biases ):
        super( Gyroscope, self ).__init__( sensor_measurement=sensor_measurement, sensitivities=sensitivities, biases=biases, multiplying_factor=1 )

    def caliberate( self ):
        pass

    def rectify_axes( self, metric_data ):
        return np.vstack( ( metric_data[ 1, : ], metric_data[ 2, : ], metric_data[ 0, : ] ) )