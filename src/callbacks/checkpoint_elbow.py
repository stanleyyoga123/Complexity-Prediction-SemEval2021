import tensorflow as tf

class BestElbow(tf.keras.callbacks.Callback):
    def __init__(self, upper_bound=0.8, filepath="model-best-elbow.h5"):
        super(BestElbow, self).__init__()
        self.best = 100
        self.upper_bound = upper_bound
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs={}):
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        diff = abs(val_loss - train_loss)

        condition = (
            (diff < self.best) and
            (val_loss <= self.upper_bound) and
            (train_loss <= self.upper_bound)
        )

        if condition:
            print(f"Take Train Epoch: {epoch}")
            self.model.save_weights(self.filepath)
            self.best = diff