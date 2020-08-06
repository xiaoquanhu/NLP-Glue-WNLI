import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from data_loader import LabelledTextDataset
import matplotlib.pyplot as plt
import numpy as np
import pickle

n_epochs = 2

dataset = dataset = LabelledTextDataset('train.tsv', 'dev.tsv')

model = torch.hub.load('huggingface/pytorch-transformers',
                       'modelForSequenceClassification', 'roberta-base')
tokenizer = torch.hub.load(
    'huggingface/pytorch-transformers', 'tokenizer', 'roberta-base')
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(dataset.df) * n_epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)

losses1 = []
losses2 = []
for i in range(n_epochs):
    train_loss = 0
    count = 0
    batch_num = 0
    model.eval()
    for x, y in dataset.get_raw_sentences(dataset.train_df):
        x = tokenizer.encode(x, add_special_tokens=True)
        x = torch.tensor([x])
        loss = model(x, labels=y)[0]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        count += 1
        train_loss += loss.item()
        batch_num += 1
        if count % 7 == 0:
            # print(f'Epoch {i} batch {batch_num} loss: {train_loss / count}')
            losses1.append(train_loss / count)
            # train_loss = 0
            # count = 0

    count = 0
    test_loss = 0
    model.eval()
    # with torch.no_grad():
    for x, y in dataset.get_raw_sentences(dataset.test_df):
        x = tokenizer.encode(x, add_special_tokens=True)
        x = torch.tensor([x])
        # print(model(x, labels=y))
        loss = model(x, labels=y)[0]
        test_loss += loss.item()
        count += 1
        if count % 7 == 0:
            losses2.append(test_loss / count)

    count += 1
    # print(f'Epoch {i} test loss: {test_loss / count}')


def plot_losses(losses):
    plt.plot(losses)
    plt.legend(['Training', 'Validation'])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, len(losses), 1))
    plt.show()


losses = [(losses1[i], losses2[i]) for i in range(len(losses1))]
plot_losses(losses)
