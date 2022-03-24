from datetime import datetime
import os
import yaml

import pandas as pd
import numpy as np

import tensorflow as tf

tf.random.set_seed(42)

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.constant import Path
from src.utility import get_config, get_latest_version, json_to_text
from src.embedder import Word2VecEmbedder, FastTextEmbedder
from src.loader import ScratchLoader
from src.regressor import ScratchRegressor
from src.evaluator import Evaluator
from src.callbacks import BestElbow


class Trainer:
    def __init__(self, config_path, embedder, prefix):
        self.config = get_config(config_path)
        self.prefix = prefix
        self.loader = ScratchLoader({**self.config["master"], **self.config["loader"]})
        self.embedder_name = embedder
        if embedder == "word2vec":
            self.embedder = Word2VecEmbedder(self.config["word2vec"])
        elif embedder == "fasttext":
            self.embedder = FastTextEmbedder(self.config["fasttext"])
        self.__init_folder()

    def __init_folder(self):
        version = get_latest_version(Path.MODEL, self.prefix, mode=self.config["mode"])
        self.folder_path = os.path.join(Path.MODEL, f"{self.prefix}-{version}")
        os.mkdir(self.folder_path)

    def fit(self):
        # Load Data
        self.data = self.loader(sample=False)

        # Train Word2Vec
        self.embedder.fit(self.data["train_embedder"])

        # Embedding config
        # embedding_config = self.embedder.get_embedding_config(self.loader.tokenizer)
        # embedding_config["trainable"] = self.config["regressor"]["trainable_embedding"]
        # embedding = self.embedder.get_layer()

        # Update Train Vectors
        self.embedder.add_tokenizer(self.loader.tokenizer)
        data_keys = ["X_train", "X_test", "X_dev"]
        type_keys = ["sentence", "token"]
        for data_key in data_keys:
            for type_key in type_keys:
                self.data[data_key][type_key] = self.embedder(self.data[data_key][type_key])

        self.model = ScratchRegressor(
            {
                # "embedding": self.embedder,
                **self.config["regressor"],
                **self.config["master"],
            }
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
                filepath=os.path.join(self.folder_path, "model-best-val-loss.h5"),
                save_best_only=True,
                save_weights_only=True,
            ),
            BestElbow(
                filepath=os.path.join(self.folder_path, "model-best-elbow.h5"),
                upper_bound=0.015,
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
            callbacks=callbacks,
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
                "truth_complexity": self.data["y_train"],
                "pred_complexity": np.array(train_pred).reshape(-1),
            }
        )
        df_dev = pd.DataFrame(
            {
                "sentence": self.data["dev"]["sentence"],
                "token": self.data["dev"]["token"],
                "truth_complexity": self.data["y_dev"],
                "pred_complexity": np.array(dev_pred).reshape(-1),
            }
        )
        df_test = pd.DataFrame(
            {
                "sentence": self.data["test"]["sentence"],
                "token": self.data["test"]["token"],
                "truth_complexity": self.data["y_test"],
                "pred_complexity": np.array(test_pred).reshape(-1),
            }
        )
        df_train["group"] = df_train["token"].apply(lambda x: len(x.split()))
        df_dev["group"] = df_dev["token"].apply(lambda x: len(x.split()))
        df_test["group"] = df_test["token"].apply(lambda x: len(x.split()))

        return df_train, df_dev, df_test

    def save(self):
        with open(os.path.join(self.folder_path, "config.yaml"), "w+") as f:
            f.write(yaml.dump(self.config))

        with open(os.path.join(self.folder_path, "notes.txt"), "w+") as f:
            msg = f"Experiment datetime: {datetime.now()}"
            f.write(msg)

        df_train, df_dev, df_test = self.evaluate()
        df_train.to_csv(os.path.join(self.folder_path, "pred_train.csv"), index=False)
        df_dev.to_csv(os.path.join(self.folder_path, "pred_dev.csv"), index=False)
        df_test.to_csv(os.path.join(self.folder_path, "pred_test.csv"), index=False)

        # Evaluate
        single_df = {
            "train": df_train[df_train["group"] == 1],
            "dev": df_dev[df_dev["group"] == 1],
            "test": df_test[df_test["group"] == 1],
        }
        multi_df = {
            "train": df_train[df_train["group"] != 1],
            "dev": df_dev[df_dev["group"] != 1],
            "test": df_test[df_test["group"] != 1],
        }

        train_eval_single = Evaluator.eval(
            single_df["train"]["truth_complexity"],
            single_df["train"]["pred_complexity"],
        )
        dev_eval_single = Evaluator.eval(
            single_df["dev"]["truth_complexity"], single_df["dev"]["pred_complexity"]
        )
        test_eval_single = Evaluator.eval(
            single_df["test"]["truth_complexity"], single_df["test"]["pred_complexity"]
        )

        train_eval_multi = Evaluator.eval(
            multi_df["train"]["truth_complexity"], multi_df["train"]["pred_complexity"]
        )
        dev_eval_multi = Evaluator.eval(
            multi_df["dev"]["truth_complexity"], multi_df["dev"]["pred_complexity"]
        )
        test_eval_multi = Evaluator.eval(
            multi_df["test"]["truth_complexity"], multi_df["test"]["pred_complexity"]
        )

        with open(os.path.join(self.folder_path, "evaluation.txt"), "w+") as f:
            msg = "Single\n"
            msg += f"Train\n{json_to_text(train_eval_single)}\n\n"
            msg += f"Dev\n{json_to_text(dev_eval_single)}\n\n"
            msg += f"Test\n{json_to_text(test_eval_single)}\n\n"

            msg += "Multi\n"
            msg += f"Train\n{json_to_text(train_eval_multi)}\n\n"
            msg += f"Dev\n{json_to_text(dev_eval_multi)}\n\n"
            msg += f"Test\n{json_to_text(test_eval_multi)}"
            f.write(msg)


def main(config, embedder, prefix):
    trainer = Trainer(config, embedder, prefix)
    trainer.fit()
    trainer.save()
