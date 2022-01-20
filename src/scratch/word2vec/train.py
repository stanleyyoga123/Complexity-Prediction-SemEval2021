import logging

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

from src.utility import get_config
from src.scratch.word2vec.embedder import Embedder
from src.scratch.word2vec.loader import Loader
from src.scratch.word2vec.classifier import Classifier


class Trainer:
    def __init__(self):
        self.config = get_config("scratch")
        self.loader = Loader(self.config["loader"])
        self.embedder = Embedder(self.config["word2vec"])

    def fit(self):
        # Load Data
        data = self.loader()

        # Train Word2Vec
        self.embedder.fit(data["train_embedder"])

        # Embedding config
        embedding_config = self.embedder.get_embedding_config(self.loader.tokenizer)
        embedding_config["trainable"] = False

        logging.info("building model")
        self.model = Classifier({"embedding": embedding_config, **self.config["classifier"]})

        self.model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=1e-3))
        self.model.fit(data["X_train"], data["y_train"], batch_size=64)

    def save(self, filepath):
        self.model.save_weights(filepath, save_format="h5")


def main():
    trainer = Trainer()
    trainer.fit()
    trainer.save("models/test.h5")
