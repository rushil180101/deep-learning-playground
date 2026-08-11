import numpy as np
from typing import Tuple
from constants import LEARNING_RATE


class RnnCell:

    def __init__(
        self,
        hidden_units: int,
        io_units: int,
        learning_rate: float = LEARNING_RATE,
    ) -> None:
        self.hidden_units = hidden_units
        self.io_units = io_units
        self.learning_rate = learning_rate
        self.stack = []
        self.init_weights()
        self.init_gradients()

    def init_weights(self) -> None:
        c = np.sqrt(1 / (self.hidden_units + self.io_units))
        # For intermediate activation
        w_aa = np.random.randn(self.hidden_units, self.hidden_units) * c
        w_ax = np.random.randn(self.hidden_units, self.io_units) * c
        self.wa = np.hstack((w_aa, w_ax))
        self.ba = np.zeros((self.hidden_units, 1))
        # For final output
        self.wy = np.random.randn(self.io_units, self.hidden_units) * c
        self.by = np.zeros((self.io_units, 1))

    def init_gradients(self) -> None:
        self.dwa = np.zeros((self.hidden_units, self.hidden_units + self.io_units))
        self.dba = np.zeros((self.hidden_units, 1))
        self.dwy = np.zeros((self.io_units, self.hidden_units))
        self.dby = np.zeros((self.io_units, 1))

    def forward(
        self,
        a_prev: np.ndarray,
        x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # First layer
        i = np.vstack((a_prev, x))
        z1 = (self.wa @ i) + self.ba
        a_next = np.tanh(z1)
        dz1 = 1 - a_next**2

        # Second layer
        z2 = (self.wy @ a_next) + self.by
        exp_z2 = np.exp(z2 - np.max(z2))  # Prevents exponentiation overflow
        sum_exp_z2 = np.sum(exp_z2)
        y_hat = exp_z2 / sum_exp_z2
        dz2 = np.diag(y_hat.flatten()) - np.outer(y_hat, y_hat)

        self.stack.append((i, a_next, dz1, dz2))
        return a_next, y_hat

    def backward(
        self,
        dy_hat: np.ndarray,
        da_future: np.ndarray,
    ) -> np.ndarray:
        i, a, dz1, dz2 = self.stack.pop()

        # Second layer
        dz = dz2 @ dy_hat
        self.dwy += dz @ a.T
        self.dby += dz

        # First layer
        da = self.wy.T @ dz
        da += da_future[: self.hidden_units]
        dz1 = da * dz1
        self.dwa += dz1 @ i.T
        self.dba += dz1
        da_prev = self.wa.T @ dz1

        return da_prev

    def update_weights(self) -> None:
        self.wa -= self.learning_rate * self.dwa
        self.ba -= self.learning_rate * self.dba
        self.wy -= self.learning_rate * self.dwy
        self.by -= self.learning_rate * self.dby
        self.init_gradients()

    def infer(
        self,
        a_prev: np.ndarray,
        x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # First layer
        i = np.vstack((a_prev, x))
        z1 = (self.wa @ i) + self.ba
        a_next = np.tanh(z1)

        # Second layer
        z2 = (self.wy @ a_next) + self.by
        exp_z2 = np.exp(z2 - np.max(z2))  # Prevents exponentiation overflow
        sum_exp_z2 = np.sum(exp_z2)
        y_hat = exp_z2 / sum_exp_z2

        return a_next, y_hat
