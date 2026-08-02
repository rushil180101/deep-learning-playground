import numpy as np
from typing import List
from rnn import RNN
from vocabulary_lib.vocabulary import BaseVocabularyManager
from logging_utils import logger

LEARNING_RATE = 1e-2
ACTIVATION_UNITS = 75
MAX_SEQUENCE_SIZE = 30
EOS_TOKEN = "<eos>"


class SequenceModel:

    def __init__(
        self,
        vocabulary_manager: BaseVocabularyManager,
        activation_units: int = ACTIVATION_UNITS,
        learning_rate: float = LEARNING_RATE,
    ) -> None:
        self.vocabulary_manager = vocabulary_manager
        self.activation_units = activation_units
        self.io_units = vocabulary_manager.get_size()
        self.learning_rate = learning_rate
        self.initialize_weights()
        self.initialize_gradients()
        self.initialize_rnns()

    def initialize_weights(self) -> None:
        c = np.sqrt(1 / self.io_units)

        # For intermediate activation
        w_aa = np.random.randn(self.activation_units, self.activation_units) * c
        w_ax = np.random.randn(self.activation_units, self.io_units) * c
        wa = np.hstack((w_aa, w_ax))
        ba = np.zeros((self.activation_units, 1))

        # For final output
        wy = np.random.randn(self.io_units, self.activation_units) * c
        by = np.zeros((self.io_units, 1))

        self.wb_params = {"wa": wa, "ba": ba, "wy": wy, "by": by}

    def initialize_gradients(self) -> None:
        self.dw_db_gradients = {
            "dwa": np.zeros(self.wb_params["wa"].shape),
            "dba": np.zeros(self.wb_params["ba"].shape),
            "dwy": np.zeros(self.wb_params["wy"].shape),
            "dby": np.zeros(self.wb_params["by"].shape),
        }

    def initialize_rnns(self) -> None:
        self.rnns = [RNN(wb_params=self.wb_params) for _ in range(MAX_SEQUENCE_SIZE)]

    def compute_cost(
        self,
        y_sequence: List[np.ndarray],
        y_hat_sequence: List[np.ndarray],
        epsilon: float = 1e-12,
    ) -> float:
        cost = 0
        for y, y_hat in zip(y_sequence, y_hat_sequence):
            c = np.argmax(y)  # index of the true class
            p = np.clip(y_hat[c], epsilon, 1 - epsilon)
            loss = -np.log(p)  # categorical cross entropy loss
            cost += loss
        return cost

    def fit(self, sentences: List[str]) -> None:

        for iteration, sentence in enumerate(sentences):
            sentence += " " + EOS_TOKEN
            sentence = sentence.strip()
            tokens = sentence.split()
            input_sequence = [
                self.vocabulary_manager.get_vector(token) for token in tokens
            ]
            input_sequence = input_sequence[
                : min(len(input_sequence), MAX_SEQUENCE_SIZE)
            ]
            last_layer_activations = []

            # Forward pass
            a = np.zeros((self.activation_units, 1))
            x = np.zeros((self.io_units, 1))
            for idx, xt in enumerate(input_sequence):
                rnn = self.rnns[idx]
                a1, a2 = rnn.forward(wb_params=self.wb_params, a=a, x=x)
                last_layer_activations.append(a2)
                a = a1
                x = xt

            # Compute cost
            cost = self.compute_cost(
                y_sequence=input_sequence,
                y_hat_sequence=last_layer_activations,
            )
            if iteration % 100 == 0:
                logger.info(f"Cost = {cost}")

            # Backward pass
            da_future = np.vstack(
                (
                    np.zeros((self.activation_units, 1)),
                    np.zeros((self.io_units, 1)),
                )
            )
            for idx in range(len(input_sequence) - 1, -1, -1):
                rnn = self.rnns[idx]
                a2 = last_layer_activations[idx]
                y = input_sequence[idx]
                da = -(y / a2)
                da_future = rnn.backward(
                    wb_params=self.wb_params,
                    dw_db_gradients=self.dw_db_gradients,
                    da_y_hat=da,
                    da_future=da_future,
                )

            # Update weights
            self.wb_params["wa"] -= self.learning_rate * self.dw_db_gradients["dwa"]
            self.wb_params["ba"] -= self.learning_rate * self.dw_db_gradients["dba"]
            self.wb_params["wy"] -= self.learning_rate * self.dw_db_gradients["dwy"]
            self.wb_params["by"] -= self.learning_rate * self.dw_db_gradients["dby"]
            self.initialize_gradients()

        logger.info(f"Iterated on {len(sentences)} sentences")

    def sample(self, count: int = 5) -> List[str]:

        sampled_sequences = []
        for _ in range(count):
            a = np.zeros((self.activation_units, 1))
            x = np.zeros((self.io_units, 1))

            sequence = []
            rnn_idx = 0
            t = ""
            while rnn_idx < len(self.rnns) and t != EOS_TOKEN:
                rnn = self.rnns[rnn_idx]
                a, y = rnn.infer(wb_params=self.wb_params, a=a, x=x)
                idx = np.random.choice(self.io_units, p=y.flatten())
                x = np.zeros((self.io_units, 1))
                x[idx, 0] = 1
                t = self.vocabulary_manager.get_token(vector=x)
                if t != EOS_TOKEN:
                    sequence.append(t)
                rnn_idx += 1

            sample_sentence = " ".join(sequence)
            sampled_sequences.append(sample_sentence)

        return sampled_sequences
