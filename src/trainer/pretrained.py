from datetime import datetime
import os
import shutil

import pandas as pd
import numpy as np

import tensorflow as tf

tf.random.set_seed(42)

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.constant import Path
from src.utility import get_config, get_latest_version, json_to_text
from src.loader import RawPretrainedLoader
from src.regressor import PretrainedRegressor
from src.evaluator import Evaluator


class Trainer:
    def __init__(self, config_path, prefix):
        self.config = get_config(config_path)
        self.prefix = prefix
        self.loader = RawPretrainedLoader(
            {**self.config["master"], **self.config["loader"]}
        )
        self.__init_folder()

    def __init_folder(self):
        version = get_latest_version(Path.MODEL, self.prefix, mode=self.config["mode"])
        self.folder_path = os.path.join(Path.MODEL, f"{self.prefix}-{version}")
        os.mkdir(self.folder_path)

    def fit(self):
        # Load Data
        self.data = self.loader()

        self.model = PretrainedRegressor(
            {
                "embedding": self.config["master"],
                **self.config["master"],
                **self.config["regressor"],
            }
        )

        if self.config["trainer"]["train_separately"]:
            self.model.embedding.trainable = False
            callbacks = [
                EarlyStopping(
                    patience=self.config["trainer"]["separate_train"][
                        "early_stopping_patience"
                    ]
                ),
                ReduceLROnPlateau(
                    factor=self.config["trainer"]["separate_train"]["reduce_lr_factor"],
                    patience=self.config["trainer"]["separate_train"][
                        "reduce_lr_patience"
                    ],
                    min_lr=self.config["trainer"]["separate_train"]["reduce_lr_min"],
                ),
                ModelCheckpoint(
                    filepath=os.path.join(self.folder_path, "model-best.h5"),
                    save_best_only=True,
                    save_weights_only=True,
                ),
            ]

            self.model.compile(
                loss=MeanSquaredError(),
                optimizer=Adam(
                    learning_rate=self.config["trainer"]["separate_train"]["lr"]
                ),
            )
            self.model.fit(
                self.data["X_train"],
                self.data["y_train"],
                batch_size=self.config["trainer"]["separate_train"]["batch_size"],
                epochs=self.config["trainer"]["separate_train"]["epochs"],
                validation_data=(self.data["X_dev"], self.data["y_dev"]),
                callbacks=callbacks,
            )
            self.model.embedding.trainable = True

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

        # Evaluate
        train_eval = Evaluator.eval(self.data["y_train"], df_train["complexity"])
        dev_eval = Evaluator.eval(self.data["y_dev"], df_dev["complexity"])
        test_eval = Evaluator.eval(self.data["y_test"], df_test["complexity"])

        with open(os.path.join(self.folder_path, "evaluation.txt"), "w+") as f:
            msg = f"Train\n{json_to_text(train_eval)}\n\n"
            msg += f"Dev\n{json_to_text(dev_eval)}\n\n"
            msg += f"Test\n{json_to_text(test_eval)}"
            f.write(msg)


def main(config, prefix):
    trainer = Trainer(config, prefix)
    trainer.fit()
    trainer.save()
