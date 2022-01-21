from datetime import datetime
import os
import shutil

import pandas as pd
import numpy as np

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.constant import Path
from src.utility import get_config, get_latest_version
from src.scratch.word2vec.embedder import Embedder
from src.scratch.word2vec.loader import Loader
from src.scratch.word2vec.classifier import Classifier


class Trainer:
    def __init__(self):
        self.config = get_config("scratch")
        self.prefix = "Word2Vec-Glove"
        self.loader = Loader(self.config["loader"])
        self.embedder = Embedder(self.config["word2vec"])
        self.__init_folder()

    def __init_folder(self):
        version = get_latest_version(Path.MODEL, self.prefix, mode=self.config["mode"])
        self.folder_path = os.path.join(Path.MODEL, f"{self.prefix}-{version}")
        os.mkdir(self.folder_path)

    def fit(self):
        # Load Data
        self.data = self.loader()

        # Train Word2Vec
        self.embedder.fit(self.data["train_embedder"])

        # Embedding config
        embedding_config = self.embedder.get_embedding_config(self.loader.tokenizer)
        embedding_config["trainable"] = self.config["classifier"]["trainable_embedding"]

        self.model = Classifier(
            {"embedding": embedding_config, **self.config["classifier"]}
        )

        # Callbacks
        callbacks = [
            EarlyStopping(patience=self.config["trainer"]["early_stopping_patience"]),
            ReduceLROnPlateau(
                factor=self.config["trainer"]["reduce_lr_factor"],
                patience=self.config["trainer"]["reduce_lr_patience"],
                min_lr=self.config["trainer"]["reduce_lr_min"],
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.folder_path, "model-best.h5"),
                save_best_only=True,
                save_weights_only=True,
            ),
        ]

        self.model.compile(
            loss=MeanSquaredError(),
            optimizer=Adam(learning_rate=self.config["trainer"]["lr"]),
        )
        self.model.fit(
            self.data["X_train"],
            self.data["y_train"],
            batch_size=self.config["trainer"]["batch_size"],
            epochs=self.config["trainer"]["epochs"],
            validation_data=(self.data["X_dev"], self.data["y_dev"]),
            callbacks=callbacks
        )

    def evaluate(self):
        train_pred = self.model.predict(
            self.data["X_train"],
            batch_size=self.config["trainer"]["batch_size"],
            verbose=1,
        )
        dev_pred = self.model.predict(
            self.data["X_dev"],
            batch_size=self.config["trainer"]["batch_size"],
            verbose=1,
        )
        test_pred = self.model.predict(
            self.data["X_test"],
            batch_size=self.config["trainer"]["batch_size"],
            verbose=1,
        )

        df_train = pd.DataFrame(
            {
                "sentence": self.data["train"]["sentence"],
                "token": self.data["train"]["token"],
                "complexity": np.array(train_pred).reshape(-1),
            }
        )
        df_dev = pd.DataFrame(
            {
                "sentence": self.data["dev"]["sentence"],
                "token": self.data["dev"]["token"],
                "complexity": np.array(dev_pred).reshape(-1),
            }
        )
        df_test = pd.DataFrame(
            {
                "sentence": self.data["test"]["sentence"],
                "token": self.data["test"]["token"],
                "complexity": np.array(test_pred).reshape(-1),
            }
        )

        return df_train, df_dev, df_test

    def save(self):
        shutil.copy2(Path.CONFIG_SCRATCH, self.folder_path)

        with open(os.path.join(self.folder_path, "notes.txt"), "w+") as f:
            msg = f"Experiment datetime: {datetime.now()}"
            f.write(msg)

        df_train, df_dev, df_test = self.evaluate()
        df_train.to_csv(os.path.join(self.folder_path, "pred_train.csv"), index=False)
        df_dev.to_csv(os.path.join(self.folder_path, "pred_dev.csv"), index=False)
        df_test.to_csv(os.path.join(self.folder_path, "pred_test.csv"), index=False)


def main():
    trainer = Trainer()
    trainer.fit()
    trainer.save()
