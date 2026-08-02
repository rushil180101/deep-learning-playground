import numpy as np
from abc import ABC, abstractmethod

EOS_TOKEN = "<eos>"


class BaseVocabularyManager(ABC):

    @abstractmethod
    def get_size(self) -> int:
        pass

    @abstractmethod
    def get_vector(self, token: str) -> np.ndarray:
        pass

    @abstractmethod
    def get_token(self, vector: np.ndarray) -> str:
        pass


class EnglishVocabularyManager(BaseVocabularyManager):

    def __init__(self) -> None:
        all_tokens = set()
        with open("vocabulary_lib/tokens.txt") as f:
            read_tokens = f.readlines()
        for read_token in read_tokens:
            all_tokens.add(read_token.strip())

        self.dict1 = {}
        self.dict2 = {}

        size = 0
        for idx, token in enumerate(all_tokens):
            self.dict1[token] = idx
            self.dict2[idx] = token
            size += 1

        self.dict1[EOS_TOKEN] = size
        self.dict2[size] = EOS_TOKEN
        size += 1
        self.size = size

    def get_size(self) -> int:
        return self.size

    def get_vector(self, token: str) -> np.ndarray:
        idx = self.dict1[token]
        vector = np.zeros((self.size, 1), dtype=int)
        vector[idx, 0] = 1
        return vector

    def get_token(self, vector: np.ndarray) -> str:
        idx = np.argmax(vector)
        token = self.dict2[idx]
        return token
