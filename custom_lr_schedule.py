import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

# Custom learning rate schedule
class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_learning_rate, decay_rate, num_layers):
        super(CustomLearningRateSchedule, self).__init__()
        self.base_learning_rate = base_learning_rate
        self.decay_rate = decay_rate
        self.num_layers = num_layers

    def __call__(self, step):
        return self.base_learning_rate / (1 + self.decay_rate * tf.cast(step, tf.float32))
    
    def get_config(self):
        return {
            'base_learning_rate': self.base_learning_rate,
            'decay_rate': self.decay_rate,
            'num_layers': self.num_layers
        }