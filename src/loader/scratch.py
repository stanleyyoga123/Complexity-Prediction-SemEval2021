import pandas as pd
import numpy as np

import tensorflow as tf
tf.random.set_seed(42)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.constant import Path
from src.features import Generator


class RawScratchLoader:
    def __init__(self, config):
        self.config = config
        if config["enhance_feat"]:
            self.generator = Generator(config["generator"])

        self.tokenizer = Tokenizer(**config["tokenizer"])
        self.pad_params = {"padding": "pre", "truncating": "post", "value": 0}

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

    def __fit(self):
        self.tokenizer.fit_on_texts(self.train["sentence"])

    def __pad(self, tokens, maxlen):
        return pad_sequences(tokens, maxlen=maxlen, **self.pad_params)

    def __convert(self, texts):
        return self.tokenizer.texts_to_sequences(texts)

    def __convert_pad(self, texts, maxlen):
        return self.__pad(self.__convert(texts), maxlen)

    def __tokenize(self):
        X_train, X_dev, X_test = {}, {}, {}

        X_train["sentence"] = self.__convert_pad(
            self.train["sentence"], self.config["sentence_maxlen"]
        )
        X_train["token"] = self.__convert_pad(
            self.train["token"], self.config["token_maxlen"]
        )

        X_dev["sentence"] = self.__convert_pad(
            self.dev["sentence"], self.config["sentence_maxlen"]
        )
        X_dev["token"] = self.__convert_pad(
            self.dev["token"], self.config["token_maxlen"]
        )

        X_test["sentence"] = self.__convert_pad(
            self.test["sentence"], self.config["sentence_maxlen"]
        )
        X_test["token"] = self.__convert_pad(
            self.test["token"], self.config["token_maxlen"]
        )

        if self.config["enhance_feat"]:
            X_train["features"] = self.generator(self.train["token"])
            X_dev["features"] = self.generator(self.dev["token"])
            X_test["features"] = self.generator(self.test["token"])

        y_train = np.array(self.train["complexity"])
        y_dev = np.array(self.dev["complexity"])
        y_test = np.array(self.test["complexity"])

        train_embedder = np.array(
            [
                sentence.split()
                for sentence in self.tokens_to_texts(
                    self.__convert(self.train["sentence"])
                )
            ]
        )

        return {
            "train_embedder": train_embedder,
            "X_train": X_train,
            "X_test": X_test,
            "X_dev": X_dev,
            "y_train": y_train,
            "y_test": y_test,
            "y_dev": y_dev,
            "train": self.train,
            "dev": self.dev,
            "test": self.test
        }

    def tokens_to_texts(self, tokens):
        return self.tokenizer.sequences_to_texts(tokens)

    def __call__(self):
        self.__drop_null()
        self.__fit()
        return self.__tokenize()
