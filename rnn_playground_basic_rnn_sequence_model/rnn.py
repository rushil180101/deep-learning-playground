import numpy as np
from typing import Tuple
from activations import Tanh, Softmax
from layer import Layer


class RNN:

    def __init__(
        self,
        wb_params: dict,
    ) -> None:
        self.layer1 = Layer(
            wb_params=wb_params,
            suffix="a",
            activation=Tanh(),
        )
        self.layer2 = Layer(
            wb_params=wb_params,
            suffix="y",
            activation=Softmax(),
        )

    def forward(
        self, wb_params: dict, a: np.ndarray, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        v_stacked = np.vstack((a, x))
        a1 = self.layer1.forward(wb_params=wb_params, suffix="a", a_prev=v_stacked)
        a2 = self.layer2.forward(wb_params=wb_params, suffix="y", a_prev=a1)
        y_hat = a2
        return a1, y_hat

    def backward(
        self,
        wb_params: dict,
        dw_db_gradients: dict,
        da_y_hat: np.ndarray,
        da_future: np.ndarray,
    ) -> np.ndarray:
        da1 = self.layer2.backward(
            wb_params=wb_params,
            dw_db_gradients=dw_db_gradients,
            suffix="y",
            da_next=da_y_hat,
        )
        # Select only the first few terms (activation size)
        # from future derivatives because we stacked vertically
        da2 = da_future[: wb_params["ba"].shape[0], :]
        da = da1 + da2
        da_prev = self.layer1.backward(
            wb_params=wb_params,
            dw_db_gradients=dw_db_gradients,
            suffix="a",
            da_next=da,
        )
        return da_prev

    def infer(
        self, wb_params: dict, a: np.ndarray, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        a0 = np.vstack((a, x))
        a1 = self.layer1.infer(wb_params=wb_params, suffix="a", a_prev=a0)
        a2 = self.layer2.infer(wb_params=wb_params, suffix="y", a_prev=a1)
        return a1, a2
