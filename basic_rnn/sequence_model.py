import numpy as np
from typing import List
from logging import Logger
from vocabulary_lib.vocabulary import BaseVocabularyManager
from rnn_cell import RnnCell
from constants import EOS_TOKEN, MAX_SEQUENCE_SIZE, SAMPLES_COUNT


class SequenceModel:

    def __init__(
        self,
        rnn_cell: RnnCell,
        vocabulary_manager: BaseVocabularyManager,
        logger: Logger,
    ) -> None:
        self.rnn_cell = rnn_cell
        self.vocabulary_manager = vocabulary_manager
        self.logger = logger

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

        count = 1
        for sentence in sentences:
            tokens = sentence.split()
            actual = [self.vocabulary_manager.get_vector(token) for token in tokens]
            predictions = []
            a_prev = np.zeros((self.rnn_cell.hidden_units, 1))
            x = np.zeros((self.rnn_cell.io_units, 1))

            # Forward pass
            for i in range(len(tokens)):
                a_next, y_hat = self.rnn_cell.forward(a_prev=a_prev, x=x)
                a_prev = a_next
                x = actual[i]
                predictions.append(y_hat)

            # Compute cost
            cost = self.compute_cost(
                y_sequence=actual,
                y_hat_sequence=predictions,
            )
            if count % 50 == 0:
                self.logger.info(f"Processing sentence {count}, Cost = {cost}")

            # Backward pass
            da_future = np.vstack(
                (
                    np.zeros((self.rnn_cell.hidden_units, 1)),
                    np.zeros((self.rnn_cell.io_units, 1)),
                )
            )
            for i in range(len(tokens) - 1, -1, -1):
                y_hat = predictions[i]
                y = actual[i]
                dy_hat = -(y / y_hat)
                da_future = self.rnn_cell.backward(dy_hat=dy_hat, da_future=da_future)

            # Update weights
            self.rnn_cell.update_weights()

            count += 1

    def sample(self, count: int = SAMPLES_COUNT) -> List[str]:

        sentences = []

        for _ in range(count):

            sentence = []
            a = np.zeros((self.rnn_cell.hidden_units, 1))
            x = np.zeros((self.rnn_cell.io_units, 1))
            t = ""
            i = 0

            while i < MAX_SEQUENCE_SIZE and t != EOS_TOKEN:
                a, y = self.rnn_cell.infer(a_prev=a, x=x)
                # Select likely index from probability distributuon
                idx = np.random.choice(range(self.rnn_cell.io_units), p=y.flatten())
                x = np.zeros((self.rnn_cell.io_units, 1))
                x[idx, 0] = 1
                t = self.vocabulary_manager.get_token(vector=x)
                if t != EOS_TOKEN:
                    sentence.append(t)

                i += 1

            sentences.append(" ".join(sentence))

        return sentences
