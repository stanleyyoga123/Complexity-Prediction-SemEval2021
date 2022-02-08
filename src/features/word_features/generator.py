from .algorithm import Algorithm
import numpy as np


class FeatureGenerator:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.algorithm = Algorithm()
        self.dictionary = {
            feat_name: self.algorithm.STR_FUNC_MAP[feat_name]
            for feat_name in feature_names
        }

    def __call__(self, texts):
        ret = {feat_name: [] for feat_name in self.feature_names}
        for words in texts:
            words = words.split()
            val = {feat_name: 0 for feat_name in self.feature_names}
            for word in words:
                for feat_name in self.feature_names:
                    val[feat_name] += self.dictionary[feat_name](word)
            for feat_name in self.feature_names:
                val[feat_name] /= len(words)
                ret[feat_name].append(val[feat_name])
        for feat_name in self.feature_names:
            ret[feat_name] = np.array(ret[feat_name])

        return ret