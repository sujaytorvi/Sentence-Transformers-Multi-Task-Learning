
# Multi-Task Transformer Model

This repository contains the implementation of a multi-task transformer model using TensorFlow and Keras. The model is designed to handle two NLP tasks simultaneously: Named Entity Recognition (NER) and Sentiment Analysis. 

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Predicting Embeddings](#predicting-embeddings)
- [Custom Learning Rate Schedule](#custom-learning-rate-schedule))
- [References](#references)

## Overview
The goal of this project is to demonstrate the implementation, training, and optimization of a multi-task learning model, particularly focusing on transformers. The model shares a common transformer backbone and has task-specific heads for NER and Sentiment Analysis.

## Requirements
To install the necessary packages, run:
```bash
pip install -r requirements.txt
```

### requirements.txt
```
tensorflow==2.12.0
numpy==1.24.3
```

## Usage
### Training the Model
1. Load sentences and labels:
    ```python
    max_length = 25  # Define max_length for padding
    texts, task_a_labels, task_b_labels = load_sentences('sample_sentences.json', max_length)
    ```

2. Tokenize and pad the sentences:
    ```python
    padded_sequences, word_index = tokenize_sentences(texts, max_length)
    ```

3. Create and compile the multi-task model:
    ```python
    vocab_size = len(word_index) + 1
    multi_task_model = create_multi_task_model(vocab_size, max_length)
    
    base_learning_rate = 0.001
    decay_rate = 0.01
    num_layers = 10
    learning_rate_schedule = CustomLearningRateSchedule(base_learning_rate, decay_rate, num_layers)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

    multi_task_model.compile(optimizer=optimizer, 
                             loss={'task_a': 'sparse_categorical_crossentropy', 'task_b': 'sparse_categorical_crossentropy'}, 
                             metrics=['accuracy'])
    ```

4. Train the model:
    ```python
    multi_task_model.fit(padded_sequences, {'task_a': task_a_labels, 'task_b': task_b_labels}, epochs=3)
    ```

### Predicting Embeddings
After training, use the model to predict and display embeddings:
```python
predictions = multi_task_model.predict(padded_sequences)
task_a_embeddings = predictions[0]
task_b_embeddings = predictions[1]

print("Embeddings for Task A (NER):", task_a_embeddings)
print("Embeddings for Task B (Sentiment):", task_b_embeddings)
```

## Custom Learning Rate Schedule
A custom learning rate schedule is used to adjust the learning rate during training:
```python
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
```


## References
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)

Feel free to raise an issue or submit a pull request if you have any suggestions or improvements!
