import tensorflow as tf

def create_optimizer(model, learning_rate=1e-5):
    # Using Adam optimizer with layer-wise learning rates
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    return optimizer
