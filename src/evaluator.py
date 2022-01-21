from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr


class Evaluator:
    @staticmethod
    def eval(y_true, y_pred):
        return {
            "pearson": pearsonr(y_true, y_pred)[0],
            "spearman": spearmanr(y_true, y_pred)[0],
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }
