import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser

unk = '<UNK>'

class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU()
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # Ensure input_vector is 2D: [1, input_dim]
        if input_vector.dim() == 1:
            input_vector = input_vector.unsqueeze(0)  # Add batch dimension

        # Obtain hidden layer representation
        hidden = self.activation(self.W1(input_vector))
        
        # Obtain output layer representation
        output_layer = self.W2(hidden)
        
        # Obtain probability distribution
        predicted_vector = self.softmax(output_layer)
        
        return predicted_vector

# Vocabulary and data loading functions as provided

def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 

def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {word: index for index, word in enumerate(vocab_list)}
    index2word = {index: word for index, word in enumerate(vocab_list)}
    vocab.add(unk)
    return vocab, word2index, index2word 

def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data

def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = [(elt["text"].split(), int(elt["stars"]-1)) for elt in training]
    val = [(elt["text"].split(), int(elt["stars"]-1)) for elt in validation]
    return tra, val

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", required=True, help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # Fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # Load data
    train_data, valid_data = load_data(args.train_data, args.val_data)
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    # Vectorize data
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)

    # Initialize model and optimizer
    model = FFNN(input_dim=len(vocab), h=args.hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training loop
    print("========== Training for {} epochs ==========".format(args.epochs))
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        correct = 0
        total = 0
        random.shuffle(train_data)
        minibatch_size = 16 
        N = len(train_data) 

        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
                loss = example_loss if loss is None else loss + example_loss
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()
        print(f"Training accuracy for epoch {epoch + 1}: {correct / total}")

    # Testing loop
    print("========== Testing Model ==========")
    with open(args.test_data) as test_f:
        test_data = json.load(test_f)
    test_data = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in test_data]
    test_data = convert_to_vector_representation(test_data, word2index)

    model.eval()
    results = []
    with torch.no_grad():
        for input_vector, gold_label in tqdm(test_data):
            predicted_vector = model(input_vector)
            predicted_label = torch.argmax(predicted_vector).item()
            results.append(predicted_label)

    # Write results to results/test.out
    os.makedirs("results", exist_ok=True)
    with open("results/test.out", "w") as f_out:
        for result in results:
            f_out.write(f"{result}\n")

    print("Results written to results/test.out")