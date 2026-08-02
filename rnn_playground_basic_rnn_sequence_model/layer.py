import numpy as np
from activations import Activation


class Layer:

    def __init__(
        self,
        wb_params: dict,
        suffix: str,
        activation: Activation,
    ) -> None:
        w = wb_params[f"w{suffix}"]
        b = wb_params[f"b{suffix}"]
        self.input_units = w.shape[1]
        self.output_units = w.shape[0]
        self.activation = activation
        self.a_prev: np.ndarray | None = None

    def forward(self, wb_params: dict, suffix: str, a_prev: np.ndarray) -> np.ndarray:
        w = wb_params[f"w{suffix}"]
        b = wb_params[f"b{suffix}"]
        self.a_prev = a_prev
        z = (w @ a_prev) + b
        a = self.activation.forward(z)
        return a

    def backward(
        self, wb_params: dict, dw_db_gradients: dict, suffix: str, da_next: np.ndarray
    ) -> np.ndarray:
        w = wb_params[f"w{suffix}"]
        b = wb_params[f"b{suffix}"]
        dz = self.activation.backward(da_next)
        dw_db_gradients[f"dw{suffix}"] += dz @ self.a_prev.T
        dw_db_gradients[f"db{suffix}"] += dz
        da_prev = w.T @ dz
        return da_prev

    def infer(self, wb_params: dict, suffix: str, a_prev: np.ndarray) -> np.ndarray:
        w = wb_params[f"w{suffix}"]
        b = wb_params[f"b{suffix}"]
        z = (w @ a_prev) + b
        a = self.activation.infer(z)
        return a
