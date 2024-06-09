import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Embedding, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model

# Create sentence transformer model
def create_sentence_transformer_model(vocab_size, max_length, embedding_dim=128, num_heads=2, ff_dim=128):
    inputs = Input(shape=(max_length,))
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)(inputs)
    transformer_block = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(embedding_layer, embedding_layer)
    transformer_block = LayerNormalization(epsilon=1e-6)(transformer_block)
    transformer_block = Dense(ff_dim, activation='relu')(transformer_block)
    transformer_block = Dense(embedding_dim)(transformer_block)
    pooling_layer = GlobalAveragePooling1D()(transformer_block)
    outputs = Dense(embedding_dim)(pooling_layer)
    model = Model(inputs=inputs, outputs=outputs)
    return model