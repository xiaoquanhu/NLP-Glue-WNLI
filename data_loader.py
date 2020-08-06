import pandas as pd
import torch
import nltk
from functools import lru_cache
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
import os


class LabelledTextDataset:
    def __init__(self, train_file_path, test_file_path):
        df1 = pd.read_table(train_file_path).sample(frac=1.0)
        df2 = pd.read_table(test_file_path).sample(frac=1.0)
        test_index = len(df1['sentence1'])
        self.df = pd.concat([df1, df2])
        # self.df.dropna(inplace=True, how='any')
        # self.df['class'] = [x.strip() for x in self.df['class']]
        # class_to_binary = {'half-true': 'true', 'true': 'true', 'false': 'false',
        #    'pants-fire': 'false', 'barely-true': 'false', 'mostly-true': 'true'}
        # self.df['class'] = [class_to_binary[c] for c in self.df['class']]

        # classes = list(set([x for x in self.df['class']]))
        # self.class_to_id = {classes[i]: i for i in range(len(classes))}
        # self.df['class'] = [self.class_to_id[c] for c in self.df['class']]
        self.df['sentences'] = [self.df['sentence1'].iloc[i] +
                                self.df['sentence2'].iloc[i] for i in range(len(self.df['sentence1']))]
        p = Preprocessor()
        self.df['tokens'] = [p(s) for s in self.df['sentences']]
        vocab = list(set((x for l in self.df['tokens'] for x in l)))
        self.token_to_id = {vocab[i]: i for i in range(len(vocab))}
        self.id_to_token = {i: vocab[i] for i in range(len(vocab))}
        # this token to id is for Glove:
        with open(os.path.join('glove.6B', 'wordtoid.pickle'), 'rb') as f:
            wordtoid = pickle.load(f)
        f.close
        self.df['ids'] = [[wordtoid[t] for t in s if t in wordtoid]
                          for s in self.df['tokens']]
        # self.df['ids'] = [[self.token_to_id[t]
        #                    for t in s] for s in self.df['tokens']]
        self.test_df = self.df.iloc[:test_index]
        self.train_df = self.df.iloc[test_index:]

    def get_ith_sentence(self, df, ind):
        s = df['ids'].iloc[ind]
        # return torch.LongTensor(s).view(-1, 1), torch.LongTensor([df['class'].iloc[ind]]).view(1)
        return s, torch.LongTensor([df['label'].iloc[ind]]).view(1)

    def get_sentences(self, df):
        return (self.get_ith_sentence(df, i) for i in range(len(df)))

    def get_raw_sentences(self, df):
        return ((self.df['sentences'].iloc[i], torch.LongTensor([df['label'].iloc[i]]).view(1)) for i in range(len(df)))


class Preprocessor:
    def __init__(self):
        self.lemma = lru_cache(maxsize=100000)(WordNetLemmatizer().lemmatize)
        self.tokenize = word_tokenize

    def __call__(self, text):
        text = text.lower()
        stopset = set(stopwords.words('english'))
        tokens = self.tokenize(text)
        tokens = [self.lemma(token)
                  for token in tokens if token not in stopset]
        return tokens
