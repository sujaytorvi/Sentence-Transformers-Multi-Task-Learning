#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries


import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

from sentence_transformer import *
from multi_task_transformer import *
from custom_lr_schedule import *


# ### Load Sentences
# 
# #### The below function `load_sentences` reads a JSON file containing sentences and their labels, extracts the texts and labels for Named Entity Recognition (NER) and sentiment analysis, and ensures the NER labels are padded to a consistent length. It finally returns the texts, padded NER labels, and sentiment labels, all ready for further processing.


# Load sentences and labels
def load_sentences(file_path, max_length=25):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    texts = []
    task_a_labels = []
    task_b_labels = []
    
    for item in data['sentences']:
        texts.append(item['text'])
        task_a_labels.append(item['task_a_ner'])
        task_b_labels.append(item['task_b_sentiment'])
    
    # Padding the NER labels to ensure all have the same length
    task_a_labels = pad_sequences(task_a_labels, maxlen=max_length, padding='post', value=0)
    task_b_labels = np.array(task_b_labels, dtype=np.int32)  # Ensure consistent data type
    
    return texts, task_a_labels, task_b_labels


# ### Tokenize Sentences
# 
# #### The below function `tokenize_sentences` converts the list of sentences into sequences of integers using a tokenizer, which maps words to unique indices. It then pads these sequences to a uniform length to ensure they all have the same size, making them ready for input into a neural network.


# Tokenize sentences
def tokenize_sentences(texts, max_length=25):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences, tokenizer.word_index


# ### Task 1: Single Task Transformer 
# 
# #### We define a sentence transformer model using Keras. We start by creating an input layer for sentences of a fixed length. Then, we add an embedding layer to convert words into dense vectors of a specified dimension. Next, we incorporate a transformer block, which includes multi-head attention and layer normalization, to capture the relationships between words. We further refine this representation with dense layers using ReLU activation, and then condense the information with a global average pooling layer. Finally, we get a fixed-length vector representation of the sentence, and the model is compiled and returned, ready for training on our sentence data.
# 
# ### Code (in sentence_transformer.py): 
# ``` python
# def create_sentence_transformer_model(vocab_size, max_length, embedding_dim=128, num_heads=2, ff_dim=128):
#     inputs = Input(shape=(max_length,))
#     embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)(inputs)
#     transformer_block = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(embedding_layer, embedding_layer)
#     transformer_block = LayerNormalization(epsilon=1e-6)(transformer_block)
#     transformer_block = Dense(ff_dim, activation='relu')(transformer_block)
#     transformer_block = Dense(embedding_dim)(transformer_block)
#     pooling_layer = GlobalAveragePooling1D()(transformer_block)
#     outputs = Dense(embedding_dim)(pooling_layer)
#     model = Model(inputs=inputs, outputs=outputs)
#     return model
# ```



# Load data and create model
max_length = 25
texts, task_a_labels, task_b_labels = load_sentences('sample_sentences.json', max_length)
padded_sequences, word_index = tokenize_sentences(texts, max_length)
vocab_size = len(word_index) + 1

model = create_sentence_transformer_model(vocab_size, max_length)
model.summary()


# ### Embeddings returned by the sentence transformer model


# Test the model
embeddings = model.predict(padded_sequences)
print("Embeddings shape:", embeddings.shape)
print("Sample embeddings:", embeddings[:2])


# ### Task 2: Multi-Task Transformer Model
# 
# #### In the multi-task transformer model, we start by defining an input layer to handle sentences of a fixed length. We then add an embedding layer to convert words into dense vectors, followed by a transformer block with multi-head attention and layer normalization to capture intricate word relationships. We refine these with dense layers, using ReLU activation, and pool the information into a fixed-length vector. This shared vector is then fed into two separate output layers: one for Named Entity Recognition (NER) and another for Sentiment Analysis. By sharing the core transformer and adding task-specific heads, we efficiently handle multiple NLP tasks in one model. It's a neat way to leverage shared knowledge across tasks while maintaining specialized outputs for each task.
# 
# ### Code (in multi_task_transformer.py)
# ``` python 
# def create_multi_task_model(vocab_size, max_length, embedding_dim=128, num_heads=2, ff_dim=128, num_classes_task_a=5, num_classes_task_b=2):
#     inputs = Input(shape=(max_length,))
#     embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)(inputs)
#     transformer_block = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(embedding_layer, embedding_layer)
#     transformer_block = LayerNormalization(epsilon=1e-6)(transformer_block)
#     transformer_block = Dense(ff_dim, activation='relu')(transformer_block)
#     transformer_block = Dense(embedding_dim)(transformer_block)
#     
#     # For NER, we use TimeDistributed to apply the dense layer to each time step
#     task_a_output = TimeDistributed(Dense(num_classes_task_a, activation='softmax'), name='task_a')(transformer_block)
#     
#     # Pooling layer for shared encoder output
#     pooling_layer = GlobalAveragePooling1D()(transformer_block)
#     
#     # Shared encoder output
#     encoded_output = Dense(embedding_dim)(pooling_layer)
#     
#     # Task B: Sentiment Analysis
#     task_b_output = Dense(num_classes_task_b, activation='softmax', name='task_b')(encoded_output)
#     
#     model = Model(inputs=inputs, outputs=[task_a_output, task_b_output])
#     return model 
# ```


max_length = 25  # Define max_length for padding
texts, task_a_labels, task_b_labels = load_sentences('sample_sentences.json', max_length)

# Tokenize and pad the sentences
padded_sequences, word_index = tokenize_sentences(texts, max_length)

# Create the multi-task model
vocab_size = len(word_index) + 1
multi_task_model = create_multi_task_model(vocab_size, max_length)



multi_task_model.compile(optimizer=optimizer, loss={'task_a': 'sparse_categorical_crossentropy', 'task_b': 'sparse_categorical_crossentropy'}, metrics=['accuracy'])



# For demonstration purposes, let's use random labels for training
task_a_labels_random = np.random.randint(0, 5, size=(len(texts), max_length))
task_b_labels_random = np.random.randint(0, 2, size=(len(texts),))

# Train the model briefly
multi_task_model.fit(padded_sequences, {'task_a': task_a_labels_random, 'task_b': task_b_labels_random}, epochs=3)

# Now we predict using the trained model
predictions = multi_task_model.predict(padded_sequences)

# Extract the embeddings for both tasks
task_a_embeddings = predictions[0]
task_b_embeddings = predictions[1]


# ### Embeddings returned by the multi-task model

# Show the embeddings
print("Embeddings for Task A (NER):")
print(task_a_embeddings)

print("\nEmbeddings for Task B (Sentiment):")
print(task_b_embeddings)

# If you want to see the embeddings for a specific input, you can index into the predictions
# For example, embeddings for the first input sentence
print("\nEmbeddings for the first input sentence (Task A):")
print(task_a_embeddings[0])

print("\nEmbeddings for the first input sentence (Task B):")
print(task_b_embeddings[0])


# ### Task 4: Layer-wise Learning Rate Implementation
# 
# #### The custom learning rate schedule class in TensorFlow is designed to dynamically adjust the learning rate during training. We start by initializing it with a base learning rate, a decay rate, and the number of layers. The __call__ method is then used to calculate the learning rate at each step by dividing the base learning rate by one plus the decay rate times the current step. This makes sure that as training progresses, the learning rate gradually decreases, helping the model converge better. We also have a get_config method to make the schedule serializable, returning the configuration parameters. This custom schedule helps us fine-tune our model's training process more effectively.
# 
# ### Code (in custom_lr_schedule.py):
# ```python
# class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#     def __init__(self, base_learning_rate, decay_rate, num_layers):
#         super(CustomLearningRateSchedule, self).__init__()
#         self.base_learning_rate = base_learning_rate
#         self.decay_rate = decay_rate
#         self.num_layers = num_layers
# 
#     def __call__(self, step):
#         return self.base_learning_rate / (1 + self.decay_rate * tf.cast(step, tf.float32))
#     
#     def get_config(self):
#         return {
#             'base_learning_rate': self.base_learning_rate,
#             'decay_rate': self.decay_rate,
#             'num_layers': self.num_layers
#         }
# ```



# Applying the custom learning rate schedule
base_learning_rate = 0.001
decay_rate = 0.01
num_layers = 10
learning_rate_schedule = CustomLearningRateSchedule(base_learning_rate, decay_rate, num_layers)
optimizer = Adam(learning_rate=learning_rate_schedule)

multi_task_model.compile(optimizer=optimizer, loss={'task_a': 'sparse_categorical_crossentropy', 'task_b': 'sparse_categorical_crossentropy'}, metrics=['accuracy'])

# Print model summary
multi_task_model.summary()


# ### Embeddings returned by the model 


# Ensure shapes and data types are correct
print(f"Shape of padded_sequences: {padded_sequences.shape}, Data type: {padded_sequences.dtype}")
print(f"Shape of task_a_labels: {task_a_labels.shape}, Data type: {task_a_labels.dtype}")
print(f"Shape of task_b_labels: {task_b_labels.shape}, Data type: {task_b_labels.dtype}")

# Train the model
multi_task_model.fit(padded_sequences, {'task_a': task_a_labels, 'task_b': task_b_labels}, epochs=30)

# Test the model
predictions = multi_task_model.predict(padded_sequences)
print("Predictions for Task A (NER):", predictions[0])
print("Predictions for Task B (Sentiment):", predictions[1])



