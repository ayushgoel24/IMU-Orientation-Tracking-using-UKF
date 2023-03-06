import numpy as np

class HelperUtils( object ):

    @staticmethod
    def get_rotation_matrix( roll: float=0, pitch: float=0, yaw: float=0 ):
        rotation_matrix = np.array([
            [ np.cos( yaw ) * np.cos( pitch ), np.cos( yaw ) * np.sin( pitch ) * np.sin( roll ) - np.sin( yaw ) * np.cos( roll ), np.cos( yaw ) * np.sin( pitch ) * np.cos( roll ) + np.sin( yaw ) * np.sin( roll ) ],
            [ np.sin( yaw ) * np.cos( pitch ), np.sin( yaw ) * np.sin( pitch ) * np.sin( roll ) + np.cos( yaw ) * np.cos( roll ), np.sin( yaw ) * np.sin( pitch ) * np.cos( roll ) - np.cos( yaw ) * np.sin( roll ) ],
            [ -1 * np.sin( pitch ), np.cos( pitch ) * np.sin( roll ), np.cos( pitch ) * np.cos( roll ) ]
        ])
        return rotation_matrix
    
    @staticmethod
    def get_quaternion_from_rotation_vector( rotation_vector ):
        alpha = np.linalg.norm( rotation_vector, axis=0 )
        alpha[ alpha==0 ] = 0.0001
        quaternion = np.zeros(( 4, rotation_vector.shape[1] ))
        quaternion[0] = np.cos( alpha / 2 )
        quaternion[1:] = rotation_vector / alpha * np.sin( alpha / 2 )
        return quaternion