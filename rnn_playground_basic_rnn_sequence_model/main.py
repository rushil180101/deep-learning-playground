"""
Within rnn_playground_1 directory, run:
python main.py
"""

from logging_utils import logger
from vocabulary_lib.vocabulary import EnglishVocabularyManager
from sequence_model import SequenceModel


def main() -> None:
    with open("corpus/file.txt") as corpus:
        sentences = corpus.readlines()

    sentences = [s.strip() for s in sentences]
    english_vocabulary_manager = EnglishVocabularyManager()
    vocabulary_size = english_vocabulary_manager.get_size()
    logger.info(f"Vocabulary size = {vocabulary_size}")
    sequence_model = SequenceModel(vocabulary_manager=english_vocabulary_manager)
    sequence_model.fit(sentences=sentences)
    logger.info("Trained RNN")

    sampled_sequences = sequence_model.sample()
    for sample_sequence in sampled_sequences:
        logger.info(f"Sample sequence = {sample_sequence}")


if __name__ == "__main__":
    main()
