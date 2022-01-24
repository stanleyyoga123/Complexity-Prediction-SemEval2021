if __name__ == "__main__":
    from src.trainer.raw.scratch import main
    main("fasttext", "raw-fasttext")
    main("word2vec", "raw-w2v")

    from src.trainer.raw.pretrained import main
    main("raw-bert")
    main("raw-xlnet")