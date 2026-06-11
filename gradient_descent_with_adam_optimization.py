"""
This module implements gradient descent with adam optimization algorithm.
Adam is a combination of the following concepts
- Exponentially weighted averages - Gradient descent with momentum
- RMSprop - Averaging over squares of last few terms

These algorithms reduce the magnitude of oscillations in gradient descent
so the gradient descent trajectory becomes smoother and converges faster
instead of the gradient descent steps getting too large in incorrect directions.
"""

import numpy as np
import sklearn.datasets
import numpy as np
from typing import List, Literal, Optional, Tuple


def load_dataset():
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=0.2)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    return train_X, train_Y


class Layer:

    def __init__(
        self,
        input_units: int,
        output_units: int,
        training_set_size: int,
        activation: Literal["relu", "tanh", "sigmoid"] = "relu",
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
    ) -> None:
        self.input_units = input_units
        self.output_units = output_units
        self.training_set_size = training_set_size
        self.activation = activation
        self.activation_mapper = {
            "relu": self.activation_relu,
            "sigmoid": self.activation_sigmoid,
            "tanh": self.activation_tanh,
        }
        self.initialize_weights()
        self.initialize_gradients(
            adam_beta1=adam_beta1, adam_beta2=adam_beta2, adam_epsilon=adam_epsilon
        )
        self.initialize_propagation_helpers()

    def initialize_weights(self) -> None:
        k = (2 if self.activation == "relu" else 1) / self.input_units
        self.w = np.random.randn(self.output_units, self.input_units) * np.sqrt(k)
        self.b = np.zeros((self.output_units, 1))

    def initialize_gradients(
        self,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
    ) -> None:
        self.dw = np.zeros((self.output_units, self.input_units))
        self.db = np.zeros((self.output_units, 1))

        # Also initialize the parameters and hyperparameters for adam optimization
        self.vdw = np.zeros(shape=self.dw.shape)
        self.vdb = np.zeros(shape=self.db.shape)
        self.sdw = np.zeros(shape=self.dw.shape)
        self.sdb = np.zeros(shape=self.db.shape)
        self.vdw_corrected = np.zeros(shape=self.dw.shape)
        self.vdb_corrected = np.zeros(shape=self.db.shape)
        self.sdw_corrected = np.zeros(shape=self.dw.shape)
        self.sdb_corrected = np.zeros(shape=self.db.shape)
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon

    def initialize_propagation_helpers(self) -> None:
        self.a_prev: Optional[np.ndarray] = None
        self.da_dz_prev: Optional[np.ndarray] = None

    def activation_relu(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a = np.maximum(0, z)
        da_dz = (z > 0).astype(np.float64)
        return a, da_dz

    def activation_sigmoid(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a = 1 / (1 + np.exp(-z))
        da_dz = a * (1 - a)
        return a, da_dz

    def activation_tanh(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a = np.tanh(z)
        da_dz = 1 - a**2
        return a, da_dz

    def backprop(self, dz_next: np.ndarray, iteration: int) -> np.ndarray:
        self.dw = (1 / self.training_set_size) * np.dot(dz_next, self.a_prev.T)
        self.db = (1 / self.training_set_size) * np.sum(dz_next, axis=1, keepdims=True)
        da_prev = np.dot(self.w.T, dz_next)
        dz_prev = da_prev * self.da_dz_prev

        # Compute adam optimization params based on latest dw and db

        # Momentum
        self.vdw = (self.adam_beta1 * self.vdw) + ((1 - self.adam_beta1) * self.dw)
        self.vdb = (self.adam_beta1 * self.vdb) + ((1 - self.adam_beta1) * self.db)

        # RMSprop
        self.sdw = (self.adam_beta2 * self.sdw) + (
            (1 - self.adam_beta2) * (self.dw) ** 2
        )
        self.sdb = (self.adam_beta2 * self.sdb) + (
            (1 - self.adam_beta2) * (self.db) ** 2
        )

        # Bias correction
        self.vdw_corrected = self.vdw / (1 - (self.adam_beta1) ** iteration)
        self.vdb_corrected = self.vdb / (1 - (self.adam_beta1) ** iteration)
        self.sdw_corrected = self.sdw / (1 - (self.adam_beta2) ** iteration)
        self.sdb_corrected = self.sdb / (1 - (self.adam_beta2) ** iteration)

        return dz_prev

    def update_weights(self, learning_rate: float = 0.01) -> None:
        w_term = self.vdw_corrected / (np.sqrt(self.sdw_corrected) + self.adam_epsilon)
        b_term = self.vdb_corrected / (np.sqrt(self.sdb_corrected) + self.adam_epsilon)
        self.w = self.w - (learning_rate * w_term)
        self.b = self.b - (learning_rate * b_term)

    def __call__(
        self, a_prev: np.ndarray, da_dz_prev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.a_prev = a_prev
        self.da_dz_prev = da_dz_prev
        z = np.dot(self.w, a_prev) + self.b
        activation_func = self.activation_mapper[self.activation]
        a, da_dz = activation_func(z)
        return a, da_dz

    def inference_step(self, a_prev: np.ndarray) -> np.ndarray:
        z = np.dot(self.w, a_prev) + self.b
        activation_func = self.activation_mapper[self.activation]
        a, _ = activation_func(z)
        return a


class NeuralNet:

    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        a_prev, da_dz_prev = x, np.ones(x.shape)
        for layer in self.layers:
            a, da_dz = layer(a_prev=a_prev, da_dz_prev=da_dz_prev)
            a_prev, da_dz_prev = a, da_dz
        return a

    def compute_cost(self, m: int, y_hat: np.ndarray, y: np.ndarray) -> float:
        eps = 1e-15
        y_hat = np.clip(y_hat, eps, 1 - eps)
        loss = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        cost = (1 / m) * np.sum(loss)
        return cost

    def backward(self, y_hat: np.ndarray, y: np.ndarray, iteration: int) -> None:
        dz_next = y_hat - y
        for layer in reversed(self.layers):
            dz = layer.backprop(dz_next=dz_next, iteration=iteration)
            dz_next = dz

    def update_nn_weights(self) -> None:
        for layer in self.layers:
            layer.update_weights()

    def predict(self, x: np.ndarray) -> np.ndarray:
        a_prev = x
        for layer in self.layers:
            a = layer.inference_step(a_prev=a_prev)
            a_prev = a
        return a

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        iterations: int = 100,
    ) -> None:
        m = x_train.shape[1]

        for it in range(1, iterations + 1):

            # Forward pass
            y_hat = self.forward(x=x_train)

            # Compute cost
            cost = self.compute_cost(m=m, y_hat=y_hat, y=y_train)
            if it % 100 == 0:
                print(f"iteration = {it}, cost = {cost}")

            # Backprop
            self.backward(y_hat=y_hat, y=y_train, iteration=it)

            # Update weights (gradient descent)
            self.update_nn_weights()

    def get_accuracy(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        y_hat = self.predict(x=x_test)
        y_hat = (y_hat >= 0.5).astype(int)
        result = (y_hat == y_test).astype(int)
        accuracy = np.sum(result) / result.shape[1]
        return accuracy


def main() -> None:

    np.random.seed(42)

    # Dataset consists of 300 examples
    x, y = load_dataset()
    total_data_points = 300
    m = 240  # 80 %
    x_train, y_train = x[:, :m], y[:, :m]
    x_test, y_test = x[:, m:], y[:, m:]

    l1 = Layer(input_units=x_train.shape[0], output_units=5, training_set_size=m)
    l2 = Layer(input_units=l1.output_units, output_units=2, training_set_size=m)
    l3 = Layer(
        input_units=l2.output_units,
        output_units=1,
        training_set_size=m,
        activation="sigmoid",
    )
    neural_net = NeuralNet(layers=[l1, l2, l3])

    neural_net.fit(x_train=x_train, y_train=y_train, iterations=1000)

    accuracy = neural_net.get_accuracy(x_test=x_test, y_test=y_test)
    print(f"accuracy = {accuracy}")


if __name__ == "__main__":
    main()
