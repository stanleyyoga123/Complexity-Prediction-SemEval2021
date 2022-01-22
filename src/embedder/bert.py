from transformers import TFBertModel


class BertEmbedder:
    def __init__(self, config):
        self.config = config
        self.embedder = TFBertModel.from_pretrained(config["model_name"])

    def __call__(self, x):
        return self.embedder(x)
