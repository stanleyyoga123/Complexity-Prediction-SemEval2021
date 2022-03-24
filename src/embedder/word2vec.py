import numpy as np
import tensorflow as tf

from gensim.models import Word2Vec
from tqdm import tqdm


class Word2VecEmbedder:
    def __init__(self, config):
        self.config = config

    def __getitem__(self, word):
        return self.embedder.wv[word]

    def fit(self, sentences):
        self.embedder = Word2Vec(sentences=sentences, **self.config)

    def get_layer(self):
        return self.embedder

    def get_embedding_config(self, tokenizer):
        weights = np.zeros((tokenizer.num_words, self.config["vector_size"]))
        for i in range(1, tokenizer.num_words):
            word = tokenizer.index_word[i]
            if word in self.embedder.wv:
                weights[i] = self.embedder.wv[word]
            else:
                print(f"Word {word} not available")

        return {
            "input_dim": tokenizer.num_words,
            "output_dim": self.config["vector_size"],
            "weights": [weights],
        }

    def add_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, inputs):
        res = []
        for tokens in inputs:
            tokens = tokens
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
                embs.append(np.array(emb))
            res.append(np.array(embs))
        return np.array(np.array(res))
