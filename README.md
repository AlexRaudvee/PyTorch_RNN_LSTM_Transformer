# Experimenting with Recurrent Neural Networks (RNNs), Long Short-Term Memory Networks (LSTMs), BI-LSTMs, and Transformers

Welcome to the repository where I experiment with various neural network architectures for tasks such as language generation, machine translation, and more. In this project, I explore state-of-the-art architectures derived from research papers provided by leading companies in the field, including OpenAI and Meta.

## Overview

The goal of this project is to understand and compare the performance of different recurrent and transformer-based models on various datasets and tasks. By implementing these architectures and applying them to tasks such as MNIST classification, language generation, and machine translation, I aim to gain insights into their strengths and weaknesses.

## Architectures Explored

### 1. Recurrent Neural Networks (RNNs)
   - Simple RNNs with feedback loops.

### 2. Long Short-Term Memory Networks (LSTMs)
   - LSTMs designed to address the vanishing gradient problem in RNNs.

### 3. Bidirectional LSTMs (BI-LSTMs)
   - LSTMs that process the input sequence in both forward and backward directions.

### 4. Transformers
   - Attention-based models that capture dependencies between input and output sequences.

## Datasets Used

I experiment with the following datasets:

- MNIST: Handwritten digit classification dataset.
- Language generation: Text corpora for generating coherent sentences. (dev)
- Machine translation: Parallel corpora for translating between different languages. (dev)

## Tasks Explored (dev)

1. **MNIST Classification**: Classifying handwritten digits into their respective classes.
2. **Language Generation**: Generating text sequences, such as sentences or paragraphs.
3. **Machine Translation**: Translating text from one language to another.

## Usage

1. Clone this repository:
   ```
    git clone https://github.com/AlexRaudvee/PyTorch_RNN_LSTM_Transformer
   ```

3. Install the required dependencies:
   ```
    pip install -r requirements.txt
   ```
