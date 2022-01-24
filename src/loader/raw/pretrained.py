import pandas as pd
import numpy as np

from transformers import BertTokenizer, XLNetTokenizer
from src.constant import Path


class RawPretrainedLoader:
    def __init__(self, config):
        self.config = config

        if config["type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["model_name"])
        elif config["type"] == "xlnet":
            self.tokenizer = XLNetTokenizer.from_pretrained(config["model_name"])
        else:
            raise ValueError("only support type (bert | xlnet)")

        self.train = pd.concat(
            [pd.read_csv(Path.TRAIN_SINGLE), pd.read_csv(Path.TRAIN_MULTI)]
        ).reset_index(drop=True)
        self.dev = pd.concat(
            [pd.read_csv(Path.DEV_SINGLE), pd.read_csv(Path.DEV_MULTI)]
        ).reset_index(drop=True)
        self.test = pd.concat(
            [pd.read_csv(Path.TEST_SINGLE), pd.read_csv(Path.TEST_MULTI)]
        ).reset_index(drop=True)

    def __drop_null(self):
        self.train = self.train.drop(self.train[self.train["token"].isnull()].index)
        self.train = self.train.reset_index(drop=True)
        self.dev = self.dev.drop(self.dev[self.dev["token"].isnull()].index)
        self.dev = self.dev.reset_index(drop=True)
        self.test = self.test.drop(self.test[self.test["token"].isnull()].index)
        self.test = self.test.reset_index(drop=True)

    def __tokenize(self):
        X_train, X_dev, X_test = {}, {}, {}
        train, dev, test = {}, {}, {}

        X_train["sentence"] = dict(
            self.tokenizer(
                list(self.train["sentence"]), **self.config["tokenizer_sentence"]
            )
        )
        X_train["token"] = dict(
            self.tokenizer(list(self.train["token"]), **self.config["tokenizer_token"])
        )

        X_dev["sentence"] = dict(
            self.tokenizer(
                list(self.dev["sentence"]), **self.config["tokenizer_sentence"]
            )
        )
        X_dev["token"] = dict(
            self.tokenizer(list(self.dev["token"]), **self.config["tokenizer_token"])
        )

        X_test["sentence"] = dict(
            self.tokenizer(
                list(self.test["sentence"]), **self.config["tokenizer_sentence"]
            )
        )
        X_test["token"] = dict(
            self.tokenizer(list(self.test["token"]), **self.config["tokenizer_token"])
        )

        y_train = np.array(self.train["complexity"])
        y_dev = np.array(self.dev["complexity"])
        y_test = np.array(self.test["complexity"])

        res = {
            "X_train": X_train,
            "X_test": X_test,
            "X_dev": X_dev,
            "y_train": y_train,
            "y_test": y_test,
            "y_dev": y_dev,
            "train": self.train,
            "dev": self.dev,
            "test": self.test,
        }

        return res

    def __call__(self):
        self.__drop_null()
        self.__tokenize()
        return self.__tokenize()
