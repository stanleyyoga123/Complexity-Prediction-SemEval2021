import numpy as np
from .word_features import FeatureGenerator
from .word_frequency import FrequencyGenerator


class Generator:
    def __init__(self, config):
        feature_names = config["feature_names"].split("|")
        list_frequency = config["list_frequency"].split("|")
        self.feature_generator = (
            FeatureGenerator(feature_names) if config["feature_names"] else None
        )
        self.frequency_generator = (
            FrequencyGenerator(list_frequency) if config["list_frequency"] else None
        )

    def __call__(self, sentences, texts):
        features = self.feature_generator(sentences, texts) if self.feature_generator else {}
        frequencies = self.frequency_generator(texts) if self.frequency_generator else {}
        new_feats = np.array(list({**features, **frequencies}.values())).T
        return new_feats
