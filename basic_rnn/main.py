"""
Usage: python main.py
"""

from logging_utils import logger
from vocabulary_lib.vocabulary import EnglishVocabularyManager
from rnn_cell import RnnCell
from sequence_model import SequenceModel
from constants import EOS_TOKEN, HIDDEN_UNITS


def main() -> None:

    english_vocabulary_manager = EnglishVocabularyManager()
    rnn_cell = RnnCell(
        hidden_units=HIDDEN_UNITS,
        io_units=english_vocabulary_manager.get_size(),
    )
    sequence_model = SequenceModel(
        rnn_cell=rnn_cell,
        vocabulary_manager=english_vocabulary_manager,
        logger=logger,
    )

    # Read and preprocess corpus
    with open("corpus/file.txt") as corpus:
        sentences = corpus.readlines()
    sentences = [s.strip() + " " + EOS_TOKEN for s in sentences]

    # Train sequence model
    sequence_model.fit(sentences=sentences)
    logger.info("Trained RNN")

    # Sample sentences
    sampled_sentences = sequence_model.sample()
    for s in sampled_sentences:
        logger.info(f"Sample = {s}")

    print("Done")


if __name__ == "__main__":
    main()
