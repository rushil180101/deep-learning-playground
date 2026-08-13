import numpy as np
from typing import List
from logging import Logger
from vocabulary_lib.vocabulary import BaseVocabularyManager
from lstm_cell import LstmCell
from constants import EOS_TOKEN, MAX_SEQUENCE_SIZE, SAMPLES_COUNT


class SequenceModel:

    def __init__(
        self,
        lstm_cell: LstmCell,
        vocabulary_manager: BaseVocabularyManager,
        logger: Logger,
    ) -> None:
        self.lstm_cell = lstm_cell
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
            c_prev = np.zeros((self.lstm_cell.hidden_units, 1))
            a_prev = np.zeros((self.lstm_cell.hidden_units, 1))
            x = np.zeros((self.lstm_cell.io_units, 1))

            # Forward pass
            for i in range(len(tokens)):
                (
                    c_prev,
                    a_prev,
                    y_hat,
                ) = self.lstm_cell.forward(c_prev=c_prev, a_prev=a_prev, x=x)
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
            d_c_future = np.zeros((self.lstm_cell.hidden_units, 1))
            d_a_future = np.zeros((self.lstm_cell.hidden_units, 1))
            for i in range(len(tokens) - 1, -1, -1):
                y_hat = predictions[i]
                y = actual[i]
                # Combined softmax + cross-entropy gradient dL/dz, where z is
                # the pre-softmax logits (wy @ a + by). This sidesteps the
                # need to differentiate softmax's full Jacobian separately.
                d_loss = y_hat - y
                (
                    d_c_future,
                    d_a_future,
                ) = self.lstm_cell.backward(
                    d_loss=d_loss,
                    d_c_future=d_c_future,
                    d_a_future=d_a_future,
                )

            # Update weights
            self.lstm_cell.update_weights()

            count += 1

    def sample(self, count: int = SAMPLES_COUNT) -> List[str]:

        sentences = []

        for _ in range(count):

            sentence = []
            c = np.zeros((self.lstm_cell.hidden_units, 1))
            a = np.zeros((self.lstm_cell.hidden_units, 1))
            x = np.zeros((self.lstm_cell.io_units, 1))
            t = ""
            i = 0

            while i < MAX_SEQUENCE_SIZE and t != EOS_TOKEN:
                c, a, y = self.lstm_cell.infer(c_prev=c, a_prev=a, x=x)
                # Select likely index from probability distributuon
                idx = np.random.choice(range(self.lstm_cell.io_units), p=y.flatten())
                x = np.zeros((self.lstm_cell.io_units, 1))
                x[idx, 0] = 1
                t = self.vocabulary_manager.get_token(vector=x)
                if t != EOS_TOKEN:
                    sentence.append(t)

                i += 1

            sentences.append(" ".join(sentence))

        return sentences
