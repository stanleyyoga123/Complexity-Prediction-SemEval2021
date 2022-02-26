import numpy as np
from gensim.models import Word2Vec


class Word2VecEmbedder:
    def __init__(self, config):
        self.config = config

    def __getitem__(self, word):
        return self.embedder.wv[word]

    def fit(self, sentences):
        self.embedder = Word2Vec(sentences=sentences, **self.config)

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
            "weights": [weights]
        }
