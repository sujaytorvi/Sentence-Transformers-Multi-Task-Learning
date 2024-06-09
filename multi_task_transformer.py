import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Embedding, MultiHeadAttention, GlobalAveragePooling1D, TimeDistributed
from tensorflow.keras.models import Model

# Create multi-task model
def create_multi_task_model(vocab_size, max_length, embedding_dim=128, num_heads=2, ff_dim=128, num_classes_task_a=5, num_classes_task_b=2):
    inputs = Input(shape=(max_length,))
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)(inputs)
    transformer_block = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(embedding_layer, embedding_layer)
    transformer_block = LayerNormalization(epsilon=1e-6)(transformer_block)
    transformer_block = Dense(ff_dim, activation='relu')(transformer_block)
    transformer_block = Dense(embedding_dim)(transformer_block)
    
    # For NER, we use TimeDistributed to apply the dense layer to each time step
    task_a_output = TimeDistributed(Dense(num_classes_task_a, activation='softmax'), name='task_a')(transformer_block)
    
    # Pooling layer for shared encoder output
    pooling_layer = GlobalAveragePooling1D()(transformer_block)
    
    # Shared encoder output
    encoded_output = Dense(embedding_dim)(pooling_layer)
    
    # Task B: Sentiment Analysis
    task_b_output = Dense(num_classes_task_b, activation='softmax', name='task_b')(encoded_output)
    
    model = Model(inputs=inputs, outputs=[task_a_output, task_b_output])
    return model
