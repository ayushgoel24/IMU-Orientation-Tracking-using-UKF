import numpy as np
from scipy import io
from quaternion import Quaternion
import math
from helper import HelperUtils
from loguru import logger
from datetime import datetime
import sys

from accelerometer import Accelerometer
from gyroscope import Gyroscope
from ukf import UnscentedKalmanFilter

#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter

def estimate_rot( data_num=1, load_vicon=False):
    #load data
    imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    if load_vicon:
        vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
    accel = imu['vals'][0:3,:]
    gyro = imu['vals'][3:6,:]
    T = np.shape(imu['ts'])[1]

    # your code goes here

    # Caliberating the accelerometer values
    accelerometer_sensitivities = [ 33.62164762778438, 33.62164762778438, 33.62164762778438 ]
    accelerometer_biases = [ 509.90857143, 501.08142857, 502.49534436 ]
    rectified_accelerometer_data = Accelerometer( sensor_measurement=accel, sensitivities=accelerometer_sensitivities, biases=accelerometer_biases ).adc_to_metric()
    # logger.debug( f"Accelerometer data shape: {rectified_accelerometer_data.shape}" )
    # logger.debug( f"Accelerometer data: {rectified_accelerometer_data}" )

    # Caliberating the gyroscope values
    gyroscope_sensitivities = [ 193.54234476, 193.54456723, 193.56434321 ]
    gyroscope_biases = [ 369.49534436, 371.49534436, 377.08142857 ]
    rectified_gyroscope_data = Gyroscope( sensor_measurement=gyro, sensitivities=gyroscope_sensitivities, biases=gyroscope_biases ).adc_to_metric()
    # logger.debug( f"Gyroscope data shape: {rectified_gyroscope_data.shape}" )
    # logger.debug( f"Gyroscope data: {rectified_gyroscope_data}" )
    
    initial_quaternion = Quaternion().q
    initial_angular_velocity = np.array([ 0.4, 0.4, 0.4 ])
    initial_estimate = np.hstack([ initial_quaternion, initial_angular_velocity ])
    # logger.debug( f"Initial estimate shape: {initial_estimate.shape}" )
    # logger.debug( f"Initial estimate: {initial_estimate}" )
    
    covariance = np.diag([ 82e-4, 82e-4, 82e-4, 33e-2, 33e-2, 4e-1 ])
    measurement_noise = np.diag([ 82e-4, 82e-4, 82e-4, 33e-2, 33e-2, 4e-1 ])
    dynamic_noise = np.diag([ 75e-1, 75e-1, 75e-1, 45e-1, 45e-1, 47e-1 ])
    # logger.debug( f"Initial covariance shape: {covariance.shape}" )
    # logger.debug( f"Initial covariance: {covariance}" )
    # logger.debug( f"Initial measurement_noise shape: {measurement_noise.shape}" )
    # logger.debug( f"Initial measurement_noise: {measurement_noise}" )
    # logger.debug( f"Initial dynamic_noise shape: {dynamic_noise.shape}" )
    # logger.debug( f"Initial dynamic_noise: {dynamic_noise}" )

    filter = UnscentedKalmanFilter( initial_estimate=initial_estimate, covariance=covariance, dynamic_noise=dynamic_noise, measurement_noise=measurement_noise, sample_size=T, time_periods=imu['ts'], accelerometer_measurements=rectified_accelerometer_data, gyroscope_measurements=rectified_gyroscope_data )
    filter.run_filter()
    
    roll = filter.roll
    pitch = filter.pitch
    yaw = filter.yaw
    filter.plot( vicon_rot=vicon['rots'], vicon_ts=vicon['ts'][0, :] )
    return roll, pitch, yaw

if __name__ == "__main__":
    currentTime = datetime.now().strftime( "%m-%d-%Y_%H:%M%S" )
    logger.add( f"logs/{currentTime}/estimate_rot.log", format="{time} {level} {message}", level="DEBUG", backtrace=True, diagnose=True )
    logger.add( sys.stdout, format="<green>{time}</green> {level} <red>{message}</red>", level="INFO", colorize=True )
    initial_quaternion = Quaternion()
    estimate_rot( load_vicon=False )
