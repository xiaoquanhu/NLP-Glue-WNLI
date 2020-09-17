import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from data_loader import LabelledTextDataset


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, dim_hidden, num_classes):
        super().__init__()
        w_matrix = torch.load(os.path.join('glove.6B', 'word_embeddings.pth'))
        self.embedder = nn.Embedding(w_matrix.shape[0], w_matrix.shape[1])
        matrix = torch.from_numpy(w_matrix)
        self.embedder.load_state_dict({'weight': matrix})
        self.gru = nn.GRU(dim_hidden, dim_hidden)
        self.W = nn.Linear(dim_hidden, num_classes)

    def forward(self, x):
        x = self.embedder(x)
        h_all, h_final = self.gru(x)
        return self.W(h_final.squeeze(0))


dataset = LabelledTextDataset('train.tsv', 'dev.tsv', external=True)
model = GRUClassifier(len(dataset.token_to_id), 50, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

losses = []
accuracies = []
for i in range(20):
    train_loss = 0
    count = 0
    batch_num = 0
    # traincorrect = 0
    for x, y in dataset.get_sentences(dataset.train_df):
        x = torch.LongTensor(x).view(-1, 1)
        yh = model(x)
        loss = F.cross_entropy(yh, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        count += 1
        batch_num += 1
        # traincorrect += (yh == y).float().sum()
        # if count % 10 == 0:
        #     print(f'Epoch {i} batch {batch_num} loss: {train_loss / count}')
        #     train_loss = 0
        #     count = 0
    average_train_loss = train_loss/count
    # train_accuracy = traincorrect / count

    test_loss = 0
    correct = 0
    count = 0
    for x, y in dataset.get_sentences(dataset.test_df):
        x = torch.LongTensor(x).view(-1, 1)
        yh = model(x)
        pred = torch.max(yh, 1)[1]
        correct += (pred == y).float().sum()
        test_loss += F.cross_entropy(yh, y)
        count += 1
    average_test_loss = test_loss/count
    losses.append((average_train_loss, average_test_loss))
    test_accuracy = correct / count
    accuracies.append(test_accuracy)

    print(f'Epoch {i} test accuracy: {test_accuracy}')


def plot_losses(losses):
    plt.plot(losses)
    plt.legend(['Training', 'Validation'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, len(losses), 1))
    plt.show()


def plot_accuracy(accuracies):
    plt.plot(accuracies)
    plt.legend(['test_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, len(accuracies), 1))
    plt.show()


plot_accuracy(accuracies)
plot_losses(losses)
