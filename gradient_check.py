"""
This module implements a small neural network with grad check on the
cat vs non-cat training set, to predict/flag cat vs non-cat images.
"""

import numpy as np
import h5py
from typing import List, Literal, Optional, Tuple


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

    def fit(
        self, x_train: np.ndarray, y_train: np.ndarray, iterations: int = 100
    ) -> None:
        m = x_train.shape[1]

        for it in range(iterations):

            # Forward pass
            y_hat = self.forward(x=x_train)

            # Compute cost
            cost = self.compute_cost(m=m, y_hat=y_hat, y=y_train)

            # Backprop
            self.backward(y_hat=y_hat, y=y_train)

            # Update weights (gradient descent)
            self.update_nn_weights()

            if it != 0 and it % 100 == 0:
                print(f"iteration = {it}, cost = {cost}")

    def predict(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        y_hat = self.forward(x=x_test)
        y_hat = (y_hat >= 0.5).astype(int)
        result = (y_hat == y_test).astype(int)
        accuracy = np.sum(result) / result.shape[1]
        return accuracy


def gradient_check(
    x: np.ndarray, y: np.ndarray, neural_net: NeuralNet, epsilon: float = 1e-7
) -> None:

    all_grads_actual = []
    all_grads_approx = []

    m = x.shape[1]
    for l in range(len(neural_net.layers)):
        w = neural_net.layers[l].w
        b = neural_net.layers[l].b

        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = w[i, j]

                # Increment
                neural_net.layers[l].w[i, j] += epsilon
                y_hat_incr = neural_net.forward(x=x)
                cost_incr = neural_net.compute_cost(m=m, y_hat=y_hat_incr, y=y)
                # Reset value
                neural_net.layers[l].w[i, j] = orig

                # Decrement
                neural_net.layers[l].w[i, j] -= epsilon
                y_hat_decr = neural_net.forward(x=x)
                cost_decr = neural_net.compute_cost(m=m, y_hat=y_hat_decr, y=y)
                # Reset value
                neural_net.layers[l].w[i, j] = orig

                # Approx grad
                grad_approx = (cost_incr - cost_decr) / (2 * epsilon)
                grad_actual = neural_net.layers[l].dw[i, j]
                all_grads_approx.append(grad_approx)
                all_grads_actual.append(grad_actual)

        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                orig = b[i, j]

                # Increment
                neural_net.layers[l].b[i, j] += epsilon
                y_hat_incr = neural_net.forward(x=x)
                cost_incr = neural_net.compute_cost(m=m, y_hat=y_hat_incr, y=y)
                # Reset value
                neural_net.layers[l].b[i, j] = orig

                # Decrement
                neural_net.layers[l].b[i, j] -= epsilon
                y_hat_decr = neural_net.forward(x=x)
                cost_decr = neural_net.compute_cost(m=m, y_hat=y_hat_decr, y=y)
                # Reset value
                neural_net.layers[l].b[i, j] = orig

                # Approx grad
                grad_approx = (cost_incr - cost_decr) / (2 * epsilon)
                grad_actual = neural_net.layers[l].db[i, j]
                all_grads_approx.append(grad_approx)
                all_grads_actual.append(grad_actual)

    all_grads_actual = np.array(all_grads_actual, dtype=np.float64)
    all_grads_approx = np.array(all_grads_approx, dtype=np.float64)

    numerator = np.linalg.norm(all_grads_actual - all_grads_approx)
    denominator = np.linalg.norm(all_grads_actual) + np.linalg.norm(all_grads_approx)
    diff = numerator / denominator
    print(f"all_grads_actual shape = {all_grads_actual.shape}")
    print(f"all_grads_approx shape = {all_grads_approx.shape}")
    print(f"diff = {diff}")
    assert diff <= 1e-5
    print("gradient check successful")


def main() -> None:

    np.random.seed(42)

    x_train, y_train, x_test, y_test = load_dataset()

    # Standardize data
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    m = x_train.shape[1]

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
    neural_net.fit(x_train=x_train, y_train=y_train, iterations=1000)

    # Test
    accuracy = neural_net.predict(x_test=x_test, y_test=y_test)
    print(f"accuracy = {accuracy}")

    # Gradient checking
    gradient_check(x=x_train, y=y_train, neural_net=neural_net)


if __name__ == "__main__":
    main()
