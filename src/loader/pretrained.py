import pandas as pd
import numpy as np

from transformers import BertTokenizer, XLNetTokenizer, RobertaTokenizer
from src.constant import Path
from src.features import Generator


class PretrainedLoader:
    def __init__(self, config):
        self.config = config
        if config["enhance_feat"]:
            self.generator = Generator(config["generator"])

        if config["type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["model_name"])
        elif config["type"] == "xlnet":
            self.tokenizer = XLNetTokenizer.from_pretrained(config["model_name"])
        elif config["type"] == "roberta":
            self.tokenizer = RobertaTokenizer.from_pretrained(config["model_name"])
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

    def __tokenize(self, sample):
        X_train, X_dev, X_test = {}, {}, {}

        if sample:
            train = self.train.sample(5)
            dev = self.dev.sample(5)
            test = self.test.sample(5)
        else:
            train = self.train
            dev = self.dev
            test = self.test

        X_train["sentence"] = dict(
            self.tokenizer(list(train["sentence"]), **self.config["tokenizer_sentence"])
        )
        X_train["token"] = dict(
            self.tokenizer(list(train["token"]), **self.config["tokenizer_token"])
        )

        X_dev["sentence"] = dict(
            self.tokenizer(list(dev["sentence"]), **self.config["tokenizer_sentence"])
        )
        X_dev["token"] = dict(
            self.tokenizer(list(dev["token"]), **self.config["tokenizer_token"])
        )

        X_test["sentence"] = dict(
            self.tokenizer(list(test["sentence"]), **self.config["tokenizer_sentence"])
        )
        X_test["token"] = dict(
            self.tokenizer(list(test["token"]), **self.config["tokenizer_token"])
        )

        if self.config["enhance_feat"]:
            X_train["features"] = self.generator(train["token"])
            X_dev["features"] = self.generator(dev["token"])
            X_test["features"] = self.generator(test["token"])

        y_train = np.array(train["complexity"])
        y_dev = np.array(dev["complexity"])
        y_test = np.array(test["complexity"])

        res = {
            "X_train": X_train,
            "X_test": X_test,
            "X_dev": X_dev,
            "y_train": y_train,
            "y_test": y_test,
            "y_dev": y_dev,
            "train": train,
            "dev": dev,
            "test": test,
        }

        return res

    def __call__(self, sample=False):
        self.__drop_null()
        return self.__tokenize(sample)
