import tensorflow as tf
import numpy as np
from tqdm import tqdm

from gensim.models import FastText


class FastTextEmbedder:
    def __init__(self, config):
        self.config = config

    def __getitem__(self, word):
        return self.embedder.wv[word]

    def fit(self, sentences):
        self.embedder = FastText(sentences=sentences, **self.config)

    def get_layer(self):
        return self.embedder

    def get_embedding_config(self, tokenizer):
        weights = np.zeros((tokenizer.num_words, self.config["vector_size"]))
        for i in range(1, self.config["vector_size"]):
            word = tokenizer.index_word[i]
            weights[i] = self.embedder.wv[word]

        return {
            "input_dim": tokenizer.num_words,
            "output_dim": self.config["vector_size"],
            "weights": [weights],
        }

    def add_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch_tokens):
        res = []
        for tokens in tqdm(batch_tokens):
            embs = []
            for token in tokens:
                try:
                    if token in self.tokenizer.index_word:
                        emb = self.embedder.wv.get_vector(
                            self.tokenizer.index_word[token]
                        )
                    else:
                        emb = np.zeros(self.config["vector_size"])
                except KeyError:
                    emb = np.zeros(self.config["vector_size"])
                embs.append(emb)
            res.append(embs)

        res = np.array(res)

        return np.array(res)
