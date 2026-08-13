import numpy as np
from typing import Tuple
from constants import LEARNING_RATE
from activations import sigmoid, softmax, tanh


class LstmCell:

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
        self._initialize_weights()
        self._initialize_gradients()

    def _initialize_weights(self) -> None:
        k = np.sqrt(1 / (self.hidden_units + self.io_units))
        rows = self.hidden_units
        cols = self.hidden_units + self.io_units

        # For candidate c~
        self.wc = np.random.randn(rows, cols) * k
        self.bc = np.zeros((rows, 1))

        # For update gate
        self.wu = np.random.randn(rows, cols) * k
        self.bu = np.zeros((rows, 1))

        # For forget gate
        self.wf = np.random.randn(rows, cols) * k
        self.bf = np.zeros((rows, 1))

        # For output gate
        self.wo = np.random.randn(rows, cols) * k
        self.bo = np.zeros((rows, 1))

        # For predictions
        self.wy = np.random.randn(self.io_units, rows) * k
        self.by = np.zeros((self.io_units, 1))

    def _initialize_gradients(self) -> None:
        rows = self.hidden_units
        cols = self.hidden_units + self.io_units

        # For candidate c~
        self.dwc = np.zeros((rows, cols))
        self.dbc = np.zeros((rows, 1))

        # For update gate
        self.dwu = np.zeros((rows, cols))
        self.dbu = np.zeros((rows, 1))

        # For forget gate
        self.dwf = np.zeros((rows, cols))
        self.dbf = np.zeros((rows, 1))

        # For output gate
        self.dwo = np.zeros((rows, cols))
        self.dbo = np.zeros((rows, 1))

        # For predictions
        self.dwy = np.zeros((self.io_units, rows))
        self.dby = np.zeros((self.io_units, 1))

    def forward(
        self,
        c_prev: np.ndarray,
        a_prev: np.ndarray,
        x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        v = np.vstack((a_prev, x))

        candidate_c, d_candidate_c = tanh((self.wc @ v) + self.bc)
        update_gate, d_update_gate = sigmoid((self.wu @ v) + self.bu)
        forget_gate, d_forget_gate = sigmoid((self.wf @ v) + self.bf)
        output_gate, d_output_gate = sigmoid((self.wo @ v) + self.bo)

        c = (update_gate * candidate_c) + (forget_gate * c_prev)
        c_activated, d_c_activated = tanh(c)
        a = output_gate * c_activated
        y_hat = softmax((self.wy @ a) + self.by)

        cache_entry = (
            c_prev,
            v,
            candidate_c,
            d_candidate_c,
            update_gate,
            d_update_gate,
            forget_gate,
            d_forget_gate,
            output_gate,
            d_output_gate,
            c_activated,
            d_c_activated,
            a,
        )
        self.stack.append(cache_entry)

        return c, a, y_hat

    def backward(
        self,
        d_loss: np.ndarray,
        d_c_future: np.ndarray,
        d_a_future: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        (
            c_prev,
            v,
            candidate_c,
            d_candidate_c,
            update_gate,
            d_update_gate,
            forget_gate,
            d_forget_gate,
            output_gate,
            d_output_gate,
            c_activated,
            d_c_activated,
            a,
        ) = self.stack.pop()

        # d_loss is expected to already be dL/dz for the softmax layer
        # (i.e. y_hat - y), the combined softmax+cross-entropy gradient
        # computed in sequence_model.py. No Jacobian multiply is needed here.
        temp = d_loss
        self.dwy += temp @ a.T
        self.dby += temp

        d_a = self.wy.T @ temp
        d_a += d_a_future

        temp = d_a * c_activated * d_output_gate
        self.dwo += temp @ v.T
        self.dbo += temp

        d_c_t = d_a * output_gate * d_c_activated
        d_c_t += d_c_future

        # Derive components from the formula: c = (update_gate * candidate_c) + (forget_gate * c_prev)

        # With respect to update gate
        temp = (d_c_t * candidate_c) * d_update_gate
        self.dwu += temp @ v.T
        self.dbu += temp

        # With respect to candidate c
        temp = (d_c_t * update_gate) * d_candidate_c
        self.dwc += temp @ v.T
        self.dbc += temp

        # With respect to forget gate
        temp = (d_c_t * c_prev) * d_forget_gate
        self.dwf += temp @ v.T
        self.dbf += temp

        # With respect to c prev
        d_c_prev = d_c_t * forget_gate

        # With respect to v (v = np.stack((a, x)))
        dv = np.zeros((self.hidden_units + self.io_units, 1))
        dv += self.wu.T @ (d_c_t * d_update_gate * candidate_c)
        dv += self.wc.T @ (d_c_t * d_candidate_c * update_gate)
        dv += self.wf.T @ (d_c_t * c_prev * d_forget_gate)
        dv += self.wo.T @ (d_a * c_activated * d_output_gate)
        d_a_prev = dv[: self.hidden_units]

        return d_c_prev, d_a_prev

    def update_weights(self) -> None:
        # For candidate c~
        self.wc -= self.learning_rate * self.dwc
        self.bc -= self.learning_rate * self.dbc

        # For update gate
        self.wu -= self.learning_rate * self.dwu
        self.bu -= self.learning_rate * self.dbu

        # For forget gate
        self.wf -= self.learning_rate * self.dwf
        self.bf -= self.learning_rate * self.dbf

        # For output gate
        self.wo -= self.learning_rate * self.dwo
        self.bo -= self.learning_rate * self.dbo

        # For predictions
        self.wy -= self.learning_rate * self.dwy
        self.by -= self.learning_rate * self.dby

        # Reset gradients
        self._initialize_gradients()

    def infer(
        self,
        c_prev: np.ndarray,
        a_prev: np.ndarray,
        x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        v = np.vstack((a_prev, x))

        candidate_c, d_candidate_c = tanh((self.wc @ v) + self.bc)
        update_gate, d_update_gate = sigmoid((self.wu @ v) + self.bu)
        forget_gate, d_forget_gate = sigmoid((self.wf @ v) + self.bf)
        output_gate, d_output_gate = sigmoid((self.wo @ v) + self.bo)

        c = (update_gate * candidate_c) + (forget_gate * c_prev)
        c_activated, d_c_activated = tanh(c)
        a = output_gate * c_activated
        y_hat = softmax((self.wy @ a) + self.by)

        return c, a, y_hat
