"""
This module implements a small neural network with grad check on the
cat vs non-cat training set, to predict/flag cat vs non-cat images.
"""

import copy
import random
import numpy as np
import h5py
from typing import Any, List, Literal, Optional, Tuple


def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    train_dataset = h5py.File("cat_vs_noncat_image_dataset/train_catvnoncat.h5", "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File("cat_vs_noncat_image_dataset/test_catvnoncat.h5", "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    nw = train_set_x_orig.shape[1]
    nh = train_set_x_orig.shape[2]
    nc = train_set_x_orig.shape[3]
    nx = nw * nh * nc

    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]

    train_set_x_orig = train_set_x_orig.reshape((m_train, nx)).T
    train_set_y_orig = train_set_y_orig.reshape((m_train, 1)).T
    test_set_x_orig = test_set_x_orig.reshape((m_test, nx)).T
    test_set_y_orig = test_set_y_orig.reshape((m_test, 1)).T

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


class Layer:

    def __init__(
        self,
        input_units: int,
        output_units: int,
        training_set_size: int,
        activation: Literal["relu", "tanh", "sigmoid"] = "relu",
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
        self.initialize_gradients()
        self.initialize_propagation_helpers()

    def initialize_weights(self) -> None:
        k = (2 if self.activation == "relu" else 1) / self.input_units
        self.w = np.random.randn(self.output_units, self.input_units) * np.sqrt(k)
        self.b = np.zeros((self.output_units, 1))

    def initialize_gradients(self) -> None:
        self.dw = np.zeros((self.output_units, self.input_units))
        self.db = np.zeros((self.output_units, 1))

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

    def backprop(self, dz_next: np.ndarray) -> np.ndarray:
        self.dw = (1 / self.training_set_size) * np.dot(dz_next, self.a_prev.T)
        self.db = (1 / self.training_set_size) * np.sum(dz_next, axis=1, keepdims=True)
        da_prev = np.dot(self.w.T, dz_next)
        dz_prev = da_prev * self.da_dz_prev
        return dz_prev

    def update_weights(self, learning_rate: float = 0.01) -> None:
        self.w = self.w - (learning_rate * self.dw)
        self.b = self.b - (learning_rate * self.db)

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

    def backward(self, y_hat: np.ndarray, y: np.ndarray) -> None:
        dz_next = y_hat - y
        for layer in reversed(self.layers):
            dz = layer.backprop(dz_next=dz_next)
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
        should_check_gradients: bool = False,
        gradient_checker: Optional[Any] = None,
    ) -> None:
        m = x_train.shape[1]

        for it in range(iterations):

            # Forward pass
            y_hat = self.forward(x=x_train)

            # Compute cost
            cost = self.compute_cost(m=m, y_hat=y_hat, y=y_train)

            # Backprop
            self.backward(y_hat=y_hat, y=y_train)

            if it != 0 and it % 100 == 0:
                print(f"iteration = {it}, cost = {cost}")
                if should_check_gradients:
                    if not gradient_checker:
                        raise Exception(
                            "gradient checking is enabled but did not receive gradient checker object"
                        )

                    gradient_checker.check(neural_net=self)

            # Update weights (gradient descent)
            self.update_nn_weights()

    def get_accuracy(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        y_hat = self.predict(x=x_test)
        y_hat = (y_hat >= 0.5).astype(int)
        result = (y_hat == y_test).astype(int)
        accuracy = np.sum(result) / result.shape[1]
        return accuracy


class GradientChecker:

    def __init__(self, x: np.ndarray, y: np.ndarray, epsilon: float = 1e-7) -> None:
        self.x = x
        self.y = y
        self.epsilon = epsilon
        self.m = x.shape[1]

    def check(self, neural_net: NeuralNet) -> None:

        # We only check for gradients on `w` weights,
        # but same thing can also be check for `b` biases

        l = random.randint(0, len(neural_net.layers) - 1)
        r = random.randint(0, neural_net.layers[l].w.shape[0] - 1)
        c = random.randint(0, neural_net.layers[l].w.shape[1] - 1)

        nn_incr = copy.deepcopy(neural_net)
        nn_incr.layers[l].w[r, c] += self.epsilon
        y_hat_incr = nn_incr.predict(x=self.x)
        cost_incr = nn_incr.compute_cost(m=self.m, y_hat=y_hat_incr, y=self.y)

        nn_decr = copy.deepcopy(neural_net)
        nn_decr.layers[l].w[r, c] -= self.epsilon
        y_hat_decr = nn_decr.predict(x=self.x)
        cost_decr = nn_decr.compute_cost(m=self.m, y_hat=y_hat_decr, y=self.y)

        grad_approx = (cost_incr - cost_decr) / (2 * self.epsilon)
        grad_actual = neural_net.layers[l].dw[r, c]

        diff = abs(grad_approx - grad_actual)
        print(f"gradient check diff = {diff}")
        assert diff <= 1e-7


def main() -> None:

    np.random.seed(42)

    x_train, y_train, x_test, y_test = load_dataset()

    # Standardize data
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    m = x_train.shape[1]

    # Gradient checker
    gradient_checker = GradientChecker(x=x_train, y=y_train)

    # Create layers
    l1 = Layer(input_units=x_train.shape[0], output_units=16, training_set_size=m)
    l2 = Layer(input_units=l1.output_units, output_units=8, training_set_size=m)
    l3 = Layer(input_units=l2.output_units, output_units=4, training_set_size=m)
    l4 = Layer(
        input_units=l3.output_units,
        output_units=1,
        training_set_size=m,
        activation="sigmoid",
    )

    # Create nn
    neural_net = NeuralNet(layers=[l1, l2, l3, l4])

    # Train
    neural_net.fit(
        x_train=x_train,
        y_train=y_train,
        iterations=1000,
        should_check_gradients=True,
        gradient_checker=gradient_checker,
    )

    # Test
    accuracy = neural_net.get_accuracy(x_test=x_test, y_test=y_test)
    print(f"accuracy = {accuracy}")


if __name__ == "__main__":
    main()
