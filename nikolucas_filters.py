import numpy as np
import pandas as pd

class AdaptiveKalmanFilter:
    def __init__(self):
        self.state_dim = 4
        self.measurement_dim = 2
        self.x = np.zeros((self.state_dim, 1))  # Initial state vector [x, y, vx, vy]
        self.P = np.eye(self.state_dim)  # Initial covariance matrix with some uncertainty
        self.F = np.eye(self.state_dim)  # State transition matrix (to be updated with dt)
        self.H = np.zeros((self.measurement_dim, self.state_dim))  # Observation matrix
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.Q = np.eye(self.state_dim) * 1e-6  # Initial process noise
        self.R = np.eye(self.measurement_dim) * 0.04  # Measurement noise covariance matrix
        self.alpha = 0.00001  # Adaptation rate for Q

    def reset(self, measurement):
        self.x[:2] = measurement[:2].reshape(2, 1)
        self.x[2:] = 0  # Initial velocities set to zero
        self.P = np.eye(self.state_dim) * 1  # Reinitialize P with some uncertainty
        return self.x[:2].flatten()

    def update(self, dt, measurement):
        # Update the state transition matrix with the new dt
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        # Prediction step
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Measurement update step
        z = measurement[:2].reshape(2, 1)
        y = z - self.H @ x_pred  # Innovation or residual
        S = self.H @ P_pred @ self.H.T + self.R  # Innovation covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = x_pred + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ P_pred

        # Adaptive process noise update
        self.Q = (1 - self.alpha) * self.Q + self.alpha * (K @ y @ y.T @ K.T)

        return self.x[:2].flatten()



"""
Was ist der Kalman Filter?
ist ein rekursiver Algorithmus, der zur Schätzung des Zustands eines dynamischen Systems verwendet

Wie funktioniert der Kalman-Filter?
Hauptsächlich in zwei Hauptschritten: dem Vorhersageschritt und dem Update-Schritt
1: Initialisierung: 
- initialen Zustandsschätzwert und initiale Kovarianzmatrix
2: Vorhersageschritt: 
Ziel: Vorhersage des nächsten Zustands des Systems basierend auf dem aktuellen Zustand und dem Modell des Systems
3: Update-Schritt: 
Ziel: korrigieren der Vorhersage basierend auf der neuen Messung
"""

state_dim = 2
measurement_dim = 2

class KalmanFilter:
    def __init__(self, state_dim):
        self.state_dim = 2
        self.measurement_dim = 2
        self.x = np.zeros((state_dim, 1))  # initialer Zustandsvektor
        self.P = np.eye(state_dim)  # initiale Kovarianzmatrix
        self.F = np.eye(state_dim)  # Zustandsübergangsmatrix
        self.H = np.eye(measurement_dim, state_dim)  # Beobachtungsmatrix
        self.Q = np.zeros((state_dim, state_dim))  # Prozessrauschen
        self.R = np.eye(measurement_dim) * 0.04  # Messrauschen

    def reset(self, measurement):
        self.x = measurement
        self.P = np.eye(self.state_dim)
        return self.x

    def update(self, dt, measurement):
        # Vorhersageschritt
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Update-Schritt
        K = P_pred @ self.H.T @ np.linalg.inv(self.H @ P_pred @ self.H.T + self.R)
        self.x = x_pred + K @ (measurement - self.H @ x_pred)
        self.P = (np.eye(self.state_dim) - K @ self.H) @ P_pred

        return self.x
    

class RandomNoise:
    def __init__(self, state_dim=2, measurement_dim=2):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.x = np.zeros((state_dim, 1))  # Initial state vector
        self.P = np.eye(state_dim) * 1.0  # Initial covariance matrix with some uncertainty
        self.F = np.eye(state_dim)  # State transition matrix
        self.H = np.eye(measurement_dim, state_dim)  # Observation matrix
        self.Q = np.eye(state_dim) * 1e-5  # Small process noise to avoid overconfidence

    def reset(self, measurement):
        self.x = measurement[:self.state_dim].reshape(self.state_dim, 1)
        self.P = np.eye(self.state_dim) * 1  # Reinitialize P with some uncertainty
        return self.x.flatten()

    def update(self, dt, measurement):
        # Extract the measured positions and the measurement noise covariance matrix
        z = measurement[:self.measurement_dim].reshape(self.measurement_dim, 1)
        Rt = measurement[2:].reshape(2, 2)
        
        # Prediction step
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Update step
        S = self.H @ P_pred @ self.H.T + Rt  # Innovation covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S + np.eye(S.shape[0]) * 1e-9)  # Kalman gain with numerical stability
        self.x = x_pred + K @ (z - self.H @ x_pred)
        self.P = (np.eye(self.state_dim) - K @ self.H) @ P_pred

        return self.x.flatten()

class ExtendedKalmanFilter:
    def __init__(self, shape):
        self.shape = shape
        self.state = np.zeros(shape)  # Initial state as 1D array
        self.P = np.eye(shape) * 10000  # Initial covariance matrix
        self.R = np.array([[0.0100, 0.0000], 
                           [0.0000, 0.0025]])  # Measurement noise covariance matrix

    def reset(self, measurement):
        # Convert polar to Cartesian coordinates for the initial state
        r, phi = measurement
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        self.state = np.array([x, y])
        self.P = np.eye(self.shape) * 10000  # Reset the covariance matrix
        return self.state

    def update(self, dt, measurement):
        # Predict step
        F = np.eye(self.shape)  # State transition matrix
        self.state = np.dot(F, self.state)  # Predict state
        self.P = np.dot(F, np.dot(self.P, F.T))  # Predict covariance

        # Measurement update step
        r, phi = measurement
        hx = np.array([np.sqrt(self.state[0]**2 + self.state[1]**2), 
                       np.arctan2(self.state[1], self.state[0])])
        
        H = self.jacobi(self.state)
        S = np.dot(H, np.dot(self.P, H.T)) + self.R
        K = np.dot(self.P, np.dot(H.T, np.linalg.inv(S)))
        
        y = np.array([r, phi]) - hx
        y[1] = self.normalize_angle(y[1])
        
        self.state = self.state + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(H, self.P))
        
        return self.state

    def jacobi(self, state):
        x, y = state
        r2 = x**2 + y**2
        r = np.sqrt(r2)
        
        H = np.array([[x/r, y/r],
                      [-y/r2, x/r2]])
        return H
    
    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

class KalmanCV():
    def __init__(self, shape):
        self.shape = shape
        self.dt = 1  # Initial time step, will be updated dynamically

        # State vector: [x_position, y_position, x_velocity, y_velocity]
        self.x = np.zeros(4)

        # State covariance matrix
        self.P = np.eye(4) * 1000

        # Process noise covariance matrix
        self.Q = np.eye(4)

        # Measurement matrix: only position is measured directly
        self.H = np.zeros((10, 4))
        for i in range(5):
            self.H[i*2:i*2+2, :2] = np.eye(2)

    def reset(self, measurement):
        # Extract initial position from the first measurement
        initial_position = np.mean(np.reshape(measurement[:10], (5, 2)), axis=0)
        self.x[:2] = initial_position  # Set initial position in the state vector
        self.x[2:] = 0  # Initial velocity is assumed to be zero
        return self.x[:2]

    def update(self, dt, measurement):
        self.dt = dt

        # State transition matrix
        F = np.eye(4)
        F[0, 2] = self.dt
        F[1, 3] = self.dt

        # Predict
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

        # Extract measurements
        z = measurement[:10]
        R = np.diag(measurement[10:])

        # Compute Kalman Gain
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.x[:2]
    
class ConstantTurnKalmanFilter():
    def __init__(self, shape):
        self.shape = shape
        self.dt = 1  # Initial time step, will be updated dynamically

        # State vector: [x_position, y_position, x_velocity, y_velocity]
        self.x = np.zeros(4)

        # State covariance matrix
        self.P = np.eye(4) * 1000

        # Process noise covariance matrix
        self.Q = np.eye(4)

        # Measurement matrix: only position is measured directly
        self.H = np.zeros((10, 4))
        for i in range(5):
            self.H[i*2:i*2+2, :2] = np.eye(2)

    def reset(self, measurement):
        # Extract initial position from the first measurement
        initial_position = np.mean(np.reshape(measurement[:10], (5, 2)), axis=0)
        self.x[:2] = initial_position  # Set initial position in the state vector
        self.x[2:] = 0  # Initial velocity is assumed to be zero
        return self.x[:2]

    def update(self, dt, measurement, turn_rate):
        self.dt = dt

        # State transition matrix with constant turn rate
        F = np.eye(4)
        F[0, 2] = self.dt
        F[1, 3] = self.dt
        
        # Rotation matrix for velocity
        theta = turn_rate * self.dt
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        # Apply rotation to velocity components
        F[2:4, 2:4] = rotation_matrix

        # Predict
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

        # Extract measurements
        z = measurement[:10]
        R = np.diag(measurement[10:])

        # Compute Kalman Gain
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.x[:2]