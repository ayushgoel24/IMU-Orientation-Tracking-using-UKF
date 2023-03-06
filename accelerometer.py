import numpy as np
from sensor import Sensor

class Accelerometer( Sensor ):

    def __init__( self, sensor_measurement, sensitivities, biases ):
        super( Accelerometer, self ).__init__( sensor_measurement=sensor_measurement, sensitivities=sensitivities, biases=biases, multiplying_factor=1 )

    def caliberate( self ):
        pass

    def rectify_axes( self, metric_data ):
        sign_rectification = np.array([ -1, -1, 1 ]).reshape( -1, 1 )
        return sign_rectification * metric_data