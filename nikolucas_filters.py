import numpy as np
import pandas as pd


class KalmanFilter:
    """
    Kalman Filter implementation for state estimation.

    Args:
        state_dim (int): Dimension of the state vector (default: 2).
        measurement_dim (int): Dimension of the measurement vector (default: 2).
    """

    def __init__(self, state_dim=2, measurement_dim=2):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.x = np.array([[1], [1]])  # initialer Zustandsvektor
        self.P = np.eye(state_dim)  # initiale Kovarianzmatrix (Anfangsunsicherheit)
        self.F = np.eye(
            state_dim
        )  # Zustandsübergangsmatrix (Vorgeschrieben: Objekt befindet sich an einem Ort)
        self.H = (
            np.eye(measurement_dim, state_dim) * 1.006
        )  # Beobachtungsmatrix (direkte Positionsmessung)
        self.Q = np.eye(state_dim) * 1e-6  # Prozessrauschen
        self.R = np.eye(measurement_dim) * 0.04  # vorgegebenes Messrauschen

    def reset(self, measurement):
        """
        Reset the filter with a new measurement.

        Args:
            measurement (ndarray): New measurement.

        Returns:
            ndarray: Updated state vector.
        """
        self.x = measurement
        self.P = np.eye(self.state_dim)
        return self.x

    def update(self, dt, measurement):
        """
        Update the filter with a new measurement.

        Args:
            dt (float): Time step.
            measurement (ndarray): New measurement.

        Returns:
            ndarray: Updated state vector.
        """
        # Vorhersageschritt
        x_pred = (
            self.F @ self.x
        )  # Zustandsschätzung basierend auf aktuellem Zustand und dem Übergang (hier konstanter Übergang)
        P_pred = (
            self.F @ self.P @ self.F.T + self.Q
        )  # Vorhersage des neuen P, basierend auf F, P und Q

        # Update-Schritt
        S = (
            self.H @ self.P @ self.H.T + self.R
        )  # stellt die Unsicherheit in der Messung dar
        K = (
            P_pred @ self.H.T @ np.linalg.inv(S)
        )  # Berechnung des Kalman-Gain, der bestimmt, wie stark die Schätzung basierend auf der neuen Messung korrigiert werden soll
        self.x = x_pred + K @ (
            measurement - self.H @ x_pred
        )  # aktualisiert die Position basierend auf der Differenz zwischen der tatsächlichen Messung und der vorhergesagten Messung
        self.P = (
            np.eye(self.state_dim) - K @ self.H
        ) @ P_pred  # Aktualisiere die Unsicherheit im neuen Zustand, basierend auf dem Kalman-Gain

        return self.x


class ConstantVelocityKalmanFilter:
    def __init__(self):
        self.state_dim = 4
        self.measurement_dim = 2  # es wird nur die Position gemessen
        self.x = np.zeros((self.state_dim, 1))  # initialer Zustand
        self.P = np.eye(self.state_dim)  # Anfangsunsicherheit
        self.F = np.eye(
            self.state_dim
        )  # konstante Geschwindigkeit = konstante Änderung der Position (weiter unten mit dt aktualisiert)
        self.H = np.zeros(
            (self.measurement_dim, self.state_dim)
        )  # Position beschreibt das System gänzlich
        self.H[0, 0] = 1
        self.H[1, 1] = 1.005
        self.Q = np.eye(self.state_dim) * 1e-8  # kleines Prozessrauschen
        self.R = np.eye(self.measurement_dim) * 0.04  # Messrauschen vorgegeben

    def reset(self, measurement):
        self.x[:2] = measurement[:2].reshape(2, 1)
        self.x[2:] = 0.0001  # Initiale Geschwindigkeit
        self.P = np.eye(self.state_dim) * 0.1  # P initialisieren
        return self.x[:2].flatten()

    def update(self, dt, measurement):
        # Update the state transition matrix with the new dt
        self.F[0, 2] = dt  # aktualisiere die Matrix
        self.F[1, 3] = dt

        # Vorhersage
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Messung updaten
        z = measurement[:2].reshape(2, 1)
        S = self.H @ P_pred @ self.H.T + self.R  # Innovationskovarianz
        K = (
            P_pred @ self.H.T @ np.linalg.inv(S)
        )  # Berechnung des Kalman-Gain (wie stark soll sich auf die Messung verlassen werden)
        self.x = x_pred + K @ (z - self.H @ x_pred)  # aktualisieren des Zustandsvekotrs
        self.P = (
            np.eye(self.state_dim) - K @ self.H
        ) @ P_pred  # neue Unsicherheitsmatrix

        return self.x[:2].flatten()


class AdaptiveKalmanFilter:
    def __init__(self):
        self.state_dim = 4
        self.measurement_dim = 2
        self.x = np.zeros((self.state_dim, 1))  # Initial state vector [x, y, vx, vy]
        self.P = np.eye(
            self.state_dim
        )  # Initial covariance matrix with some uncertainty
        self.F = np.eye(
            self.state_dim
        )  # State transition matrix (to be updated with dt)
        self.H = np.zeros((self.measurement_dim, self.state_dim))  # Observation matrix
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.Q = np.eye(self.state_dim) * 1e-7  # Initial process noise
        self.R = (
            np.eye(self.measurement_dim) * 0.04
        )  # Measurement noise covariance matrix
        self.alpha = 0.0000000001  # Adaptation rate for Q

    def reset(self, measurement):
        self.x[:2] = measurement[:2].reshape(2, 1)
        self.x[2:] = 0  # Initial velocities set to zero
        self.P = np.eye(self.state_dim)  # Reinitialize P with some uncertainty
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


class RandomNoise:
    def __init__(self, state_dim=2, measurement_dim=2):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.x = np.zeros((state_dim, 1))  # Ausgangszustand
        self.P = np.eye(state_dim)  # Anfangsunsicherheit
        self.F = np.eye(state_dim)  # Position des Objektes bleibt gleich
        self.H = (
            np.eye(measurement_dim, state_dim) * 1.001
        )  # wieder direkte Positionsmessung
        self.Q = np.eye(state_dim) * 1e-9  # kleines Prozessrauschen

    def reset(self, measurement):
        self.x = measurement[: self.state_dim].reshape(
            self.state_dim, 1
        )  # Position wird auf basierend auf der Messung initialisiert
        self.P = np.eye(self.state_dim)  # Anfangsunsicherheit initialisieren
        return self.x.flatten()

    def update(self, dt, measurement):
        # Positionsmessungen und Messrauschenmatrix extrahieren
        z = measurement[: self.measurement_dim].reshape(
            self.measurement_dim, 1
        )  # extrahiert die Positionen der Messungen
        Rt = measurement[2:].reshape(2, 2)  # extrahiert das Messrauschen

        # Prädiktionsschritt
        x_pred = (
            self.F @ self.x
        )  # Zustandsschätzung basierend auf aktuellem Zustand und dem Übergang (hier konstanter Übergang)
        P_pred = (
            self.F @ self.P @ self.F.T + self.Q
        )  # Vorhersage des neuen P, basierend auf F, P und Q

        # Update Schritt
        S = self.H @ P_pred @ self.H.T + Rt  # berechnet die Innovationskovarianz
        K = (
            P_pred @ self.H.T @ np.linalg.inv(S + np.eye(S.shape[0]) * 1e-9)
        )  # berechnet das Kalman-Gain (bestimmt, wie stark die Messung korrigiert werden soll)
        self.x = x_pred + K @ (z - self.H @ x_pred)  # neue Zustandsschätzung
        self.P = (np.eye(self.state_dim) - K @ self.H) @ P_pred  # neue Kovarianzmatrix

        return self.x.flatten()


class ExtendedKalmanFilter:
    def __init__(
        self, process_noise=1e-5, measurement_noise_r=0.01, measurement_noise_phi=0.0025
    ):
        self.state_dim = 2
        self.measurement_dim = 2
        self.x = np.zeros((self.state_dim, 1))  # initialer Zustandsvektor
        self.P = np.eye(self.state_dim)  # initiale Kovarianzmatrix
        self.Q = (
            np.eye(self.state_dim) * process_noise
        )  # Kovarianzmatrix mit geringem Prozessrauschen
        self.R = np.array(
            [[measurement_noise_r, 0], [0, measurement_noise_phi]]
        )  # Messrauschen-Kovarianzmatrix

    def reset(self, measurement):
        r, phi = measurement
        self.x[0] = r * np.cos(phi)
        self.x[1] = r * np.sin(phi)
        self.P = np.eye(self.state_dim)  # Unsicherheit aktualisieren
        return self.x.flatten()

    def h(self, x):  # Funktion um den Zustand in die Messung umzuwandeln
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        phi = np.arctan2(x[1], x[0])
        return np.array([r, phi]).reshape(-1, 1)

    def jacobi(self, x):
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        H = np.zeros((self.measurement_dim, self.state_dim))
        H[0, 0] = x[0] / r
        H[0, 1] = x[1] / r
        H[1, 0] = -x[1] / (r**2)
        H[1, 1] = x[0] / (r**2)
        return H

    def update(self, dt, measurement):
        # Prediction step (no movement, so prediction is the same as previous state)
        x_pred = self.x
        P_pred = self.P + self.Q

        # Update der
        z = measurement.reshape(self.measurement_dim, 1)
        h_x = self.h(x_pred)
        y = z - h_x
        H = self.jacobi(x_pred)
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ P_pred

        return self.x.flatten()


class ConstantVelocity2:
    def __init__(self):
        self.state_dim = 4  # [x, y, vx, vy]
        self.measurement_dim = (
            10  # Five 2D position measurements [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
        )
        self.x = np.zeros((self.state_dim, 1))  # Initial state
        self.P = np.eye(self.state_dim)  # Initial covariance matrix
        self.F = np.eye(
            self.state_dim
        )  # State transition matrix (to be updated with dt)
        self.Q = (
            np.eye(self.state_dim) * 1e-8
        )  # Small process noise to avoid overconfidence
        self.H = np.zeros((self.measurement_dim, self.state_dim))  # Observation matrix
        self.R = np.eye(
            self.measurement_dim
        )  # Measurement noise covariance matrix (to be updated with R from measurement)

        # observation matrix for the state
        for i in range(5):
            self.H[2 * i, 0] = 1
            self.H[2 * i + 1, 1] = 1

    def reset(self, measurement):
        self.x[:2] = np.mean(measurement[:10].reshape(5, 2), axis=0).reshape(
            2, 1
        )  # Initial position as the mean of measurements
        self.x[2:] = 0.001  # Initial velocities set to a small value
        self.P = np.eye(self.state_dim)  # Reinitialize P with some uncertainty
        return self.x[:2].flatten()

    def update(self, dt, measurement):
        # Update the state transition matrix with the new dt
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        # Prediction step
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Extract the measured positions and the measurement noise covariance matrix
        # extrahiere die Position und Messungs
        z = measurement[:10].reshape(self.measurement_dim, 1)
        Rt = np.diag(measurement[10:])

        # Update the measurement noise covariance matrix
        self.R = Rt

        # Measurement update step
        S = self.H @ P_pred @ self.H.T + self.R  # Innovation covariance
        K = (
            P_pred @ self.H.T @ np.linalg.inv(S + np.eye(S.shape[0]) * 1e-9)
        )  # Kalman gain with numerical stability
        self.x = x_pred + K @ (z - self.H @ x_pred)
        self.P = (np.eye(self.state_dim) - K @ self.H) @ P_pred

        return self.x[:2].flatten()


class ConstantTurnRateFilter:
    """
    A class representing a constant turn rate filter.

    Attributes:
        state_dim (int): The dimension of the state vector.
        turn_rate (float): The turn rate should be in radians per time step, but we found the best value as a float.
        window_size (int): The size of the moving average window.

    Methods:
        __init__(self, state_dim, turn_rate=0.001, window_size=5): Initializes the filter.
        reset(self, measurement): Resets the filter with a measurement.
        predict(self, dt): Predicts the state of the filter.
        update(self, dt, measurement): Updates the state of the filter.

    """

    def __init__(self, state_dim, turn_rate=0.001, window_size=5):
        """
        Initializes the constant turn rate filter.

        Args:
            state_dim (int): The dimension of the state vector.
            turn_rate (float, optional): The turn rate in radians per time step. Defaults to 0.001.
            window_size (int, optional): The size of the moving average window. Defaults to 5.

        """
        self.state_dim = state_dim
        self.turn_rate = turn_rate  # Turn rate in radians per time step
        self.state = None
        self.covariance = None
        self.process_noise = np.diag([1e-3, 1e-3, 1e-2, 1e-2])  # Tuning process noise
        self.window_size = window_size
        self.predictions = []

    def reset(self, measurement):
        """
        Resets the filter with a measurement.

        Args:
            measurement (array-like): The measurement used to reset the filter.

        Returns:
            array-like: The initial position of the filter.

        """
        x_init = np.mean(measurement[:10:2])
        y_init = np.mean(measurement[1:10:2])
        self.state = np.array([x_init, y_init, 0, 0])  # Initial position, zero velocity
        self.covariance = np.eye(self.state_dim) * 1e3  # High initial uncertainty
        self.predictions = [
            self.state[:2]
        ] * self.window_size  # Initialize the moving average window
        return self.state[:2]  # Return only the position

    def predict(self, dt):
        """
        Predicts the state of the filter.

        Args:
            dt (float): The time step.

        Returns:
            array-like: The predicted state of the filter.

        """
        theta = self.turn_rate * dt
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # State transition matrix incorporating turn rate
        F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, cos_theta, -sin_theta],
                [0, 0, sin_theta, cos_theta],
            ]
        )

        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + self.process_noise
        return self.state

    def update(self, dt, measurement):
        """
        Updates the state of the filter.

        Args:
            dt (float): The time step.
            measurement (array-like): The measurement used to update the filter.

        Returns:
            array-like: The smoothed position of the filter.

        """
        H = np.zeros((10, self.state_dim))
        for i in range(5):
            H[2 * i, 0] = 1
            H[2 * i + 1, 1] = 1

        R = np.diag(measurement[10:] ** 2)

        # Predict step
        self.predict(dt)

        # Measurement prediction
        z_pred = H @ self.state

        # Innovation or measurement residual
        y = measurement[:10] - z_pred

        # Innovation (or residual) covariance
        S = H @ self.covariance @ H.T + R

        # Optimal Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Update state estimate
        self.state = self.state + K @ y

        # Update the state covariance matrix
        I = np.eye(self.state_dim)
        self.covariance = (I - K @ H) @ self.covariance

        # Add the current position to the moving average window
        self.predictions.append(self.state[:2])
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)

        # Return the smoothed position (average of the window)
        smoothed_position = np.mean(self.predictions, axis=0)
        return smoothed_position
