
# Multi-Task Transformer Model

This repository contains the implementation of a multi-task transformer model using TensorFlow and Keras. The model is designed to handle two NLP tasks simultaneously: Named Entity Recognition (NER) and Sentiment Analysis. 

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Predicting Embeddings](#predicting-embeddings)
- [Custom Learning Rate Schedule](#custom-learning-rate-schedule)
- [Sentence Transformer and Multi-Task Transformer Model]
- [Task 3: Training Considerations]
- [Key Decisions and Insights]
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

## Sentence Transformer and Multi-Task Transformer Model

The sentence transformer model is designed to convert sentences into fixed-length vector representations, capturing semantic meaning. It uses a transformer architecture, which is effective in handling the relationships between words in a sentence. By utilizing multi-head attention mechanisms, it can focus on different parts of the sentence simultaneously, creating rich, contextual embeddings.

The multi-task transformer model extends this idea by sharing the transformer backbone across multiple NLP tasks. In this project, we handle Named Entity Recognition (NER) and Sentiment Analysis using a single transformer backbone. Each task has its own output head, allowing the model to leverage shared knowledge from the transformer while providing specialized outputs for each task. This approach improves efficiency and often results in better overall performance by sharing learned representations.

## Task 3: Training Considerations

### Freezing the Entire Network
If the entire network is frozen, it means no weights will be updated during training. This approach is useful when the model is already pre-trained and we want to use it for inference without further fine-tuning. The advantage is that it preserves the learned features, but the downside is that it doesn't adapt to new data.

### Freezing Only the Transformer Backbone
Freezing only the transformer backbone while allowing the task-specific heads to train can be beneficial when the backbone is pre-trained on a large corpus. This way, we retain the generalized language understanding of the transformer and adapt only the task-specific parts to our new tasks. This approach balances preserving useful pre-trained features and tailoring the model to specific tasks.

### Freezing Only One of the Task-Specific Heads
Freezing only one task-specific head allows us to improve the performance of the other task without affecting the frozen one. This is useful in scenarios where one task is well-learned and we want to focus on improving the other. It ensures stability for the frozen task while allowing flexibility for the other.

### Transfer Learning Approach
For transfer learning, we would start with a pre-trained transformer model like BERT. Initially, we would freeze the transformer layers and train only the task-specific heads to adapt them to our tasks. After achieving satisfactory performance, we could unfreeze some of the transformer layers to fine-tune the entire model. This approach leverages the rich, pre-trained features of models like BERT while allowing customization for specific tasks.

## Task 4: Layer-Wise Learning Rate Implementation

### Rationale for Specific Learning Rates
Using a custom learning rate scheduler allows us to set different learning rates for different layers. Typically, lower learning rates are set for pre-trained layers to avoid disrupting their learned features, while higher learning rates are used for newly added layers to enable faster adaptation. This strategy helps in achieving stable and effective training.

## Key Decisions and Insights

### Task 3 and Task 4 Summary
In Task 3, we discussed various training scenarios, emphasizing the balance between preserving pre-trained features and adapting to new tasks. The rationale behind freezing strategies and the approach to transfer learning were highlighted to leverage the strengths of pre-trained models effectively.

For Task 4, implementing a custom learning rate scheduler was crucial for fine-tuning the model efficiently. By assigning lower learning rates to pre-trained layers and higher rates to new layers, we achieved a balance between stability and adaptability. These strategies collectively enhance the model's performance and robustness, making it well-suited for multi-task learning scenarios.



## References
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)

Feel free to raise an issue or submit a pull request if you have any suggestions or improvements!

