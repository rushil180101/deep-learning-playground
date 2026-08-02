import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):

    def __init__(self) -> None:
        self.da_dz: np.ndarray | None = None

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def infer(self, z: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, da_next: np.ndarray) -> np.ndarray:
        pass


class Tanh(Activation):

    def forward(self, z: np.ndarray) -> np.ndarray:
        a = np.tanh(z)
        self.da_dz = 1 - a**2
        return a

    def infer(self, z: np.ndarray) -> np.ndarray:
        a = np.tanh(z)
        return a

    def backward(self, da_next: np.ndarray) -> np.ndarray:
        dz = da_next * self.da_dz
        return dz


class Relu(Activation):

    def forward(self, z: np.ndarray) -> np.ndarray:
        a = np.maximum(0, z)
        self.da_dz = (z > 0).astype(np.float64)
        return a

    def infer(self, z: np.ndarray) -> np.ndarray:
        a = np.maximum(0, z)
        return a

    def backward(self, da_next: np.ndarray) -> np.ndarray:
        dz = da_next * self.da_dz
        return dz


class Softmax(Activation):

    def forward(self, z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z))  # Prevents exponentiation overflow
        sum_exp_z = np.sum(exp_z)
        a = exp_z / sum_exp_z
        self.da_dz = np.diag(a.flatten()) - np.outer(a, a)  # Jacobian
        return a

    def infer(self, z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z))  # Prevents exponentiation overflow
        sum_exp_z = np.sum(exp_z)
        a = exp_z / sum_exp_z
        return a

    def backward(self, da_next: np.ndarray) -> np.ndarray:
        dz = self.da_dz @ da_next
        return dz
