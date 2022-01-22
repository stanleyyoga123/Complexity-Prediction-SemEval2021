# from src.trainer.raw.scratch import main
from src.trainer.raw.pretrained import main

if __name__ == "__main__":
    main("raw-bert")
    # main("fasttext", "raw-fasttext")
    # main("word2vec", "raw-w2v")