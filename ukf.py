from filter import Filter
import math
import numpy as np
from quaternion import Quaternion
from helper import HelperUtils
import scipy
from loguru import logger
import matplotlib.pyplot as plt

class UnscentedKalmanFilter( Filter ):

    def __init__(self, initial_estimate: np.array, covariance: np.array, dynamic_noise: np.array, measurement_noise: np.array, sample_size: int, time_periods: np.array, accelerometer_measurements: np.array, gyroscope_measurements: np.array ) -> None:
        super( UnscentedKalmanFilter, self ).__init__( initial_estimate=initial_estimate, covariance=covariance, dynamic_noise=dynamic_noise, measurement_noise=measurement_noise, time_periods=time_periods, accelerometer_measurements=accelerometer_measurements, gyroscope_measurements=gyroscope_measurements )
        self.sample_size = sample_size

        self.roll = list()
        self.pitch = list()
        self.yaw = list()
        self.estimated_states = np.zeros(( 7, sample_size ))
        self.estimated_covariances = np.zeros(( 6, 6, sample_size ))

        # Initialising
        self.estimated_states[ :, 0 ] = initial_estimate
        self.estimated_covariances[ :, :, 0 ] = covariance
        euler_angles_from_inital_quaternion = Quaternion( scalar=initial_estimate[0], vec=initial_estimate[1:4] ).euler_angles()
        
        self.roll.append( euler_angles_from_inital_quaternion[0] )
        self.pitch.append( euler_angles_from_inital_quaternion[1] )
        self.yaw.append( euler_angles_from_inital_quaternion[2] )

    def compute_and_transform_sigma_points( self, index: int ) -> np.ndarray:
        S = math.sqrt( self.estimated_covariances.shape[0] ) * scipy.linalg.sqrtm( self.estimated_covariances[ :, :, index - 1 ] + self.measurement_noise )
        Wi = np.hstack(( S , -1 * S ))
        sigma_points = np.zeros(( 7, Wi.shape[1] ))
        for i in range( Wi.shape[1] ):
            quaternion_from_axis_angle = Quaternion()
            quaternion_from_axis_angle.from_axis_angle( a=Wi[ :3, i ].reshape(-1) )
            previous_quaternion = Quaternion( scalar=self.estimated_states[ 0, index - 1 ], vec=self.estimated_states[ 1:4, index - 1 ] )

            quaternion_after_rotation = previous_quaternion.__mul__( quaternion_from_axis_angle )
            transformed_quaternion = Quaternion()

            transformed_quaternion.from_axis_angle( a=self.estimated_states[ 4:, index - 1 ] * ( self.time_periods[0, index] - self.time_periods[0, index-1] ))
            quaternion_after_transformation = quaternion_after_rotation.__mul__( transformed_quaternion )

            sigma_points[ :4, i ] = quaternion_after_transformation.q
            sigma_points[ 4:, i ] = self.estimated_states[ 4:, index - 1 ] + Wi[ 3:, i ]
        return sigma_points
    
    def gradient_descent_for_mean_covariance( self, quaternions, initial_mean_quaternion ):
        mean_quaternion = initial_mean_quaternion
        threshold = 0.01
        max_interations = 100
        errors = np.zeros(( quaternions.shape[1], 3 ))
        execution_count = 0
        while True:
            for q in range( quaternions.shape[1] ):
                quaternion_q = Quaternion( scalar=quaternions[ 0, q ], vec=quaternions[ 1:4, q ] )

                quaternion_relative = quaternion_q.__mul__( mean_quaternion.inv() )
                quaternion_relative = Quaternion( scalar=1, vec=quaternion_relative.vec()) if ( quaternion_relative.scalar() - 1.0 > 0.0  and abs( quaternion_relative.scalar() - 1.0 ) < 0.0001 ) else quaternion_relative
                quaternion_relative = Quaternion( scalar=-1, vec=quaternion_relative.vec()) if ( quaternion_relative.scalar() + 1.0 < 0.0 and abs( quaternion_relative.scalar() + 1.0 ) < 0.0001 ) else quaternion_relative
                errors[ q, : ] = quaternion_relative.axis_angle()


            mean_error = np.mean( errors, axis=0 )
            mean_quaternion_from_errors = Quaternion()
            mean_quaternion_from_errors.from_axis_angle( a=mean_error )

            mean_quaternion = mean_quaternion_from_errors.__mul__( mean_quaternion )

            if ( np.all( abs( mean_error ) < threshold ) or execution_count > max_interations ):
                break
            execution_count += 1
        
        return mean_quaternion, errors
    
    def compute_covariance( self, Wi ):
        return ( Wi.T @ Wi ) / Wi.shape[0 ] 
    
    def compute_mean_variance_from_sigma_points( self, sigma_points ):
        initial_quaternion = Quaternion( scalar=sigma_points[ 0, 0 ], vec=sigma_points[ 1:4, 0 ].flatten() )
        mean_quaternion, errors = self.gradient_descent_for_mean_covariance( quaternions=sigma_points[ :4, : ] , initial_mean_quaternion=initial_quaternion )

        mean_angular_velocities_of_sigma_points = np.mean( sigma_points[ 4:, :], axis=1 )
        Wi = sigma_points[ 4:, :] - mean_angular_velocities_of_sigma_points[:, np.newaxis] * np.ones(( 3, sigma_points.shape[1] )) 
        mean_covariance = self.compute_covariance( np.hstack(( errors, Wi.T )))

        return mean_quaternion, mean_covariance, errors, Wi
    
    def apply_measurement_model( self, measurement ):
        q_k = Quaternion( scalar=measurement[0], vec=measurement[1:4] )
        g = Quaternion( scalar=0, vec=[ 0, 0, 9.81 ])
        g_dash = q_k.inv().__mul__( g.__mul__( q_k ) )
        return np.hstack(( g_dash.vec(), measurement[4:] ))

    def compute_cross_correlation( self, stacked_errors, Zi ):
        return ( stacked_errors @ Zi.T ) / stacked_errors.shape[1]
    
    def measurement_update( self, previous_estimate, covariance, measurements, sigma_points, stacked_errors ):
        Zi = np.zeros(( 6, sigma_points.shape[1] ))
        for i in range( sigma_points.shape[1] ):
            Zi[ :, i ] = self.apply_measurement_model( sigma_points[ :, i ] )
        # logger.debug( f"Zi shape: {Zi.shape}" )
        # logger.debug( f"Zi: {Zi}" )

        z_mean = np.mean( Zi, axis=1 )[:, np.newaxis]
        # logger.debug( f"z_mean shape: {z_mean.shape}" )
        # logger.debug( f"z_mean: {z_mean}" )

        Pzz = ( ( Zi - z_mean ) @ ( Zi - z_mean ).T ) / 12
        # logger.debug( f"Pzz shape: {Pzz.shape}" )
        # logger.debug( f"Pzz: {Pzz}" )

        Pvv = Pzz + self.dynamic_noise
        # logger.debug( f"Pvv shape: {Pvv.shape}" )
        # logger.debug( f"Pvv: {Pvv}" )

        Pxz = self.compute_cross_correlation( stacked_errors=stacked_errors, Zi=Zi - z_mean )
        # logger.debug( f"Pxz shape: {Pxz.shape}" )
        # logger.debug( f"Pxz: {Pxz}" )
        
        kalmanGain = Pxz @ np.linalg.inv( Pvv )
        # logger.debug( f"kalmanGain shape: {kalmanGain.shape}" )
        # logger.debug( f"kalmanGain: {kalmanGain}" )
        
        innovation = measurements - np.squeeze( z_mean )
        # logger.debug( f"innovation shape: {innovation.shape}" )
        # logger.debug( f"innovation: {innovation}" )

        gain = kalmanGain @ innovation
        # logger.debug( f"gain shape: {gain.shape}" )
        # logger.debug( f"gain: {gain}" )
        
        quaternion_gain = Quaternion()
        quaternion_gain.from_axis_angle( a=gain[ :3 ] )
        # logger.debug( f"quaternion_gain: {quaternion_gain.__str__()}" )

        update_of_posteriori_estimate = quaternion_gain.__mul__( other=Quaternion( scalar=previous_estimate[0], vec=previous_estimate[ 1:4 ] ) )
        # logger.debug( f"update_of_posteriori_estimate: {update_of_posteriori_estimate.__str__()}" )

        new_estimate = np.hstack(( update_of_posteriori_estimate.q, previous_estimate[4:] + gain[ 3:6 ] ))
        new_covariance = covariance - kalmanGain @ Pvv @ kalmanGain.T

        return ( new_estimate, new_covariance )

    def propagate_dynamics_by_one_timestep(self, dt: float) -> None:
        return super().propagate_dynamics_by_one_timestep(dt)

    def run_filter( self ):

        for k in range( 1, self.sample_size ):
            # logger.debug( f"Iteration: {k}" )
            
            # compute sigma points
            sigma_points = self.compute_and_transform_sigma_points( index=k )
            # logger.debug( f"sigma_points shape: {sigma_points.shape}" )
            # logger.debug( f"sigma_points: {sigma_points}" )

            # compute propagated mean and covariance
            mean_quaternion, mean_covariance, errors, Wi = self.compute_mean_variance_from_sigma_points( sigma_points )
            # logger.debug( f"mean_quaternion: {mean_quaternion.__str__()}" )
            # logger.debug( f"mean_covariance shape: {mean_covariance.shape}" )
            # logger.debug( f"mean_covariance: {mean_covariance}" )
            # logger.debug( f"errors shape: {errors.shape}" )
            # logger.debug( f"errors: {errors}" )
            # logger.debug( f"Wi shape: {Wi.shape}" )
            # logger.debug( f"Wi: {Wi}" )

            measurements = np.hstack(( self.accelerometer_measurements[ :, k-1], self.gyroscope_measurements[ :, k-1 ] ))
            # logger.debug( f"measurements shape: {measurements.shape}" )
            # logger.debug( f"measurements: {measurements}" )

            new_estimate, new_covariance = self.measurement_update( np.hstack(( mean_quaternion.q, np.mean( sigma_points[ 4:, :], axis=1 ) )), mean_covariance, measurements, sigma_points, np.vstack(( errors.T, Wi )) )
            # logger.debug( f"new_estimate shape: {new_estimate.shape}" )
            # logger.debug( f"new_estimate: {new_estimate}" )
            # logger.debug( f"new_covariance shape: {new_covariance.shape}" )
            # logger.debug( f"new_covariance: {new_covariance}" )

            quaternion_from_new_estimate = Quaternion( scalar=new_estimate[0], vec=new_estimate[1:4] )
            euler_angles_from_new_estimate = quaternion_from_new_estimate.euler_angles()

            self.roll.append( euler_angles_from_new_estimate[0] )
            self.pitch.append( euler_angles_from_new_estimate[1] )
            self.yaw.append( euler_angles_from_new_estimate[2] )

            self.estimated_states[ :, k ] = new_estimate
            self.estimated_covariances[ :, :, k ] = new_covariance

    def convert_to_euler( self, vicon_rot ):
        vicon_euler = []
        vicon_quat = []
        vicon_axis_angle = []
        q = Quaternion()
        for i in range(vicon_rot.shape[2]):
            rot = vicon_rot[:, :, i]
            q.from_rotm(rot)
            euler = q.euler_angles()
            # euler = Rotation.from_matrix(rot).as_euler("XYZ")
            vicon_quat.append(q.q)
            vicon_euler.append(euler)
            vicon_axis_angle.append(q.axis_angle())
        return np.array(vicon_euler).T, np.array(vicon_quat).T, np.array(vicon_axis_angle).T

    def convert_x_axis_angle( self, xhat):
        x = []
        for i in range(0, xhat.shape[1]):
            q = Quaternion(xhat[0, i], xhat[1:4, i])
            out = q.axis_angle()
            x.append(out)
        return np.array(x)
    
    def plot( self, roll=None, pitch=None, yaw=None, gyro_corrected=None, imu_t=None, xhat=None, P=None, vicon_rot=None, vicon_ts=None ):
        roll = self.roll
        pitch = self.pitch
        yaw = self.yaw
        gyro_corrected = self.gyroscope_measurements
        imu_t = self.time_periods
        # xhat = self.estimated_states[ :, -1 ]
        # P = self.estimated_covariances[ :, :, -1 ]
        xhat = self.estimated_states
        P = self.estimated_covariances

        plt.figure()
        vicon_euler, vicon_quat, vicon_axis_angle = self.convert_to_euler(vicon_rot)

        n = np.min([len(vicon_ts), imu_t.shape[1]])
        print ("n: ", n, imu_t.shape)
        plt.plot(vicon_ts[:n], vicon_euler[0, :n], label='Roll from Vicon')
        plt.plot(np.squeeze(imu_t)[0:n], roll[0:n], '--', label='Roll from UKF')
        # plt.plot(imu_t[0:n], roll[0:n] + P[:n, 0, 0], '--', label='ukf roll + std_dev')
        # plt.plot(imu_t[0:n], roll[0:n] - P[:n, 0, 0], '--', label='ukf roll - std_dev')
        plt.legend(loc='best')
        plt.xlabel("timestamp (s)")
        plt.ylabel("Roll angle (radians)")
        plt.title("Actual value vs Predicted value")
        plt.show()
        #
        plt.figure()
        plt.plot(vicon_ts[:n], vicon_euler[1, :n], label='Pitch from Vicon')
        plt.plot(np.squeeze(imu_t)[0:n], pitch[0:n], '--', label='Pitch from UKF')
        # plt.plot(imu_t[0:n], pitch[0:n] + P[:n, 1, 1],'--', label='ukf pitch + std_dev')
        # plt.plot(imu_t[0:n], pitch[0:n] - P[:n, 1, 1], '--', label='ukf pitch - std_dev')
        plt.legend(loc='best')
        plt.xlabel("timestamp (s)")
        plt.ylabel("Pitch angle (radians)")
        plt.title("Actual value vs Predicted value")
        plt.show()
        #
        plt.figure()
        plt.plot(vicon_ts[:n], vicon_euler[2, :n], label='Yaw from Vicon')
        plt.plot(np.squeeze(imu_t)[0:n], yaw[:n], '--', label='Yaw from UKF')
        # plt.plot(imu_t[0:n], yaw[0:n] + P[:n, 2, 2], '--', label='ukf yaw + std_dev')
        # plt.plot(imu_t[0:n], yaw[0:n] - P[:n, 2, 2], '--', label='ukf yaw - std_dev')
        plt.legend(loc='best')
        plt.xlabel("timestamp (s)")
        plt.ylabel("Yaw angle (radians)")
        plt.title("Actual value vs Predicted value")
        plt.show()
        #
        plt.figure()
        plt.plot(np.squeeze(imu_t)[:n], gyro_corrected[0, :n], label='Raw angular velocity in X')
        plt.plot(np.squeeze(imu_t)[:n], xhat[4, :n], '--', label='Corrected angular velocity in X')
        # plt.plot(np.squeeze(imu_t)[:n], xhat[4, :n] + P[ 3, 3, :n], '--', label='Imu filtered w_x + sigma')
        # plt.plot(np.squeeze(imu_t)[:n], xhat[4, :n] - P[ 3, 3, :n], '--', label='Imu filtered w_x - sigma')
        plt.legend(loc='best')
        plt.xlabel("timestamp (s)")
        plt.ylabel("Angular velocity in X (rad/s)")
        plt.title("Raw vs corrected angular velocity in X")
        plt.show()
        #
        plt.figure()
        plt.plot(np.squeeze(imu_t)[:n], gyro_corrected[1, :n], label='Raw angular velocity in Y')
        plt.plot(np.squeeze(imu_t)[:n], xhat[5, :n], '--', label='Corrected angular velocity in Y')
        # plt.plot(np.squeeze(imu_t)[:n], xhat[5, :n] + P[ 4, 4, :n], '--', label='Imu filtered w_y + sigma')
        # plt.plot(np.squeeze(imu_t)[:n], xhat[5, :n] - P[ 4, 4, :n], '--', label='Imu filtered w_y - sigma')
        plt.legend(loc='best')
        plt.xlabel("timestamp (s)")
        plt.ylabel("Angular velocity in Y (rad/s)")
        plt.title("Raw vs corrected angular velocity in Y")
        plt.show()
        #
        plt.figure()
        plt.plot(np.squeeze(imu_t)[:n], gyro_corrected[2, :n], label='Raw angular velocity in Z')
        plt.plot(np.squeeze(imu_t)[:n], xhat[6, :n], '--', label='Corrected angular velocity in Z')
        # plt.plot(np.squeeze(imu_t)[:n], xhat[6, :n] + P[ 5, 5, :n], '--', label='Imu filtered w_z + sigma')
        # plt.plot(np.squeeze(imu_t)[:n], xhat[6, :n] - P[ 5, 5, :n], '--', label='Imu filtered w_z - sigma')
        plt.legend(loc='best')
        plt.xlabel("timestamp (s)")
        plt.ylabel("Angular velocity in Z (rad/s)")
        plt.title("Raw vs corrected angular velocity in Z")
        plt.show()

        # upd1, upd2 = convert_cov_to4d(xhat, P)
        # print (upd1[:n].shape, upd2[:n].shape, xhat.shape, imu_t[:n].shape)
        plt.figure()
        plt.plot(np.squeeze(imu_t)[:n], vicon_quat[0, :n], label='Scalar of Vicon quaternion')
        plt.plot(np.squeeze(imu_t)[:n], xhat[0, :n], label='Predicted quaternion scalar component')
        # plt.plot(imu_t[:n], upd1[0, :n], '--', label='predicted quat scalar with +std_dev')
        # plt.plot(imu_t[:n], upd2[0, :n], '--', label='predicted quat scalar with -std_dev')
        plt.legend(loc='best')
        plt.xlabel("timestamp (s)")
        plt.ylabel("Quaternion components")
        plt.title("Actual vs Predicted value of scalar component of Quaternion")
        plt.show()

        plt.figure()
        plt.plot(np.squeeze(imu_t)[:n], vicon_quat[1, :n], label='X of Vicon quaternion')
        plt.plot(np.squeeze(imu_t)[:n], xhat[1, :n], label='Predicted X of quaternion')
        plt.legend(loc='best')
        plt.xlabel("timestamp (s)")
        plt.ylabel("X component of Quaternion")
        plt.title("Actual vs Predicted value of X component of Quaternion")
        plt.show()

        plt.figure()
        plt.plot(np.squeeze(imu_t)[:n], vicon_quat[2, :n], label='Y of Vicon quaternion')
        plt.plot(np.squeeze(imu_t)[:n], xhat[2, :n], label='Predicted Y of quaternion')
        plt.legend(loc='best')
        plt.xlabel("timestamp (s)")
        plt.ylabel("Y component of Quaternion")
        plt.title("Actual vs Predicted value of Y component of Quaternion")
        plt.show()

        plt.figure()
        plt.plot(np.squeeze(imu_t)[:n], vicon_quat[3, :n], label='Z of Vicon quaternion')
        plt.plot(np.squeeze(imu_t)[:n], xhat[3, :n], label='Predicted Z of quaternion')
        plt.legend(loc='best')
        plt.xlabel("timestamp (s)")
        plt.ylabel("Z component of Quaternion")
        plt.title("Actual vs Predicted value of Z component of Quaternion")
        plt.show()