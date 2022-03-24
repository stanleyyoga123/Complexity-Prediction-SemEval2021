from datetime import datetime
import os
import yaml

import pandas as pd
import numpy as np

import tensorflow as tf

tf.random.set_seed(42)

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateScheduler,
)

from src.constant import Path
from src.utility import get_config, get_latest_version, json_to_text
from src.loader import PretrainedLoader
from src.regressor import PretrainedRegressor
from src.evaluator import Evaluator
from src.callbacks import BestElbow

# from src.metrics import pearson


def scheduler(epoch, lr):
    return lr * tf.math.exp(-0.1)


class Trainer:
    def __init__(self, config_path, prefix):
        self.config = get_config(config_path)
        self.prefix = prefix
        self.loader = PretrainedLoader(
            {**self.config["master"], **self.config["loader"]}
        )
        self.__init_folder()

    def __init_folder(self):
        version = get_latest_version(Path.MODEL, self.prefix, mode=self.config["mode"])
        self.folder_path = os.path.join(Path.MODEL, f"{self.prefix}-{version}")
        os.mkdir(self.folder_path)

    def fit(self):
        # Load Data
        self.data = self.loader(sample=False)
        
        # Set GPU
        devices = tf.config.experimental.list_physical_devices("GPU")
        device_names = [d.name.split("e:")[1] for d in devices]
        config_taken = set([int(i) for i in self.config["trainer"]["gpus"].split("|")])
        taken_gpu = []
        for i, device_name in enumerate(device_names):
            if i in config_taken:
                taken_gpu.append(device_name)
        print(f"Taken GPU: {taken_gpu}")
        strategy = tf.distribute.MirroredStrategy(devices=taken_gpu)

        with strategy.scope():
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
                    LearningRateScheduler(scheduler),
                    ModelCheckpoint(
                        filepath=os.path.join(self.folder_path, "model-best-val-loss.h5"),
                        save_best_only=True,
                        save_weights_only=True,
                    ),
                    BestElbow(
                        filepath=os.path.join(self.folder_path, "model-best-elbow.h5"),
                        upper_bound=0.008,
                    )
                ]

                self.model.compile(
                    loss=MeanSquaredError(),
                    optimizer=Adam(
                        learning_rate=self.config["trainer"]["separate_train"]["lr"],
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
                self.model.summary()
                self.model.embedding.trainable = True

        callbacks = [
            EarlyStopping(patience=self.config["trainer"]["early_stopping_patience"]),
            LearningRateScheduler(scheduler),
            ModelCheckpoint(
                filepath=os.path.join(self.folder_path, "model-best.h5"),
                save_best_only=True,
                save_weights_only=True,
            ),
            BestElbow(
                filepath=os.path.join(self.folder_path, "model-best-elbow.h5")
            )
        ]
        with strategy.scope():
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

        self.model.save_weights(os.path.join(self.folder_path, "model-last.h5"))

    def evaluate(self):
        # Load best weights
        self.model = PretrainedRegressor(
            {
                "embedding": self.config["master"],
                **self.config["master"],
                **self.config["regressor"],
            }
        )
        sample = self.loader(sample=True)
        self.model(sample["X_train"])
        self.model.load_weights(os.path.join(self.folder_path, "model-best-elbow.h5"))

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


def main(config, prefix):
    trainer = Trainer(config, prefix)
    trainer.fit()
    trainer.save()
