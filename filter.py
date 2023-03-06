import numpy as np

class Filter( object ):

    def __init__(self, initial_estimate: np.array, covariance: np.array, dynamic_noise: np.array, measurement_noise: np.array, time_periods: np.array, accelerometer_measurements: np.array, gyroscope_measurements: np.array ) -> None:
        self.initial_estimate = initial_estimate
        self.covariance = covariance
        self.dynamic_noise = dynamic_noise
        self.measurement_noise = measurement_noise
        self.time_periods = time_periods
        self.accelerometer_measurements = accelerometer_measurements
        self.gyroscope_measurements = gyroscope_measurements

    def propagate_dynamics_by_one_timestep(self, dt: float) -> None:
        pass