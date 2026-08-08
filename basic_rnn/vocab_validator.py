"""
Within rnn_playground_basic_rnn_sequence_model directory, run:
python vocab_validator.py
"""


def main() -> None:
    with open("corpus/file.txt") as corpus:
        sentences = corpus.readlines()
    with open("vocabulary_lib/tokens.txt") as vocab_file:
        tokens = vocab_file.readlines()

    sentences = [s.strip() for s in sentences]
    tokens = [t.strip() for t in tokens]

    for sentence in sentences:
        for token in sentence.split():
            if token not in tokens:
                with open("vocabulary_lib/tokens.txt", "a") as f:
                    f.write(f"{token}\n")
                    print(f"Written {token}")


if __name__ == "__main__":
    main()
