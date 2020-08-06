import os
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics
import nltk
from functools import lru_cache
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import torch
import matplotlib.pyplot as plt


class LabelledSentenceDF:
    def __init__(self, directory, preprocessor):
        self.df = pd.read_table(directory).sample(frac=1.0)
        self.df['sentences'] = [self.df['sentence1'].iloc[i] +
                                self.df['sentence2'].iloc[i] for i in range(len(self.df['sentence1']))]
        self.apply_preprocessor(preprocessor)
        # vocabulary = set([t for se in self.df['tokenized'] for t in se])
        # print(directory, 'vocal size', len(vocabulary))

    def apply_preprocessor(self, preprocessor):
        self.df['tokenized'] = [preprocessor(s) for s in self.df['sentences']]


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


traindata = LabelledSentenceDF('train.tsv', Preprocessor())
index = len(traindata.df['sentences'])
data = pd.concat(
    [traindata.df, LabelledSentenceDF('dev.tsv', Preprocessor()).df])

vectorizer = CountVectorizer(lowercase=False,
                             tokenizer=lambda x: x,  # tokenization has already been done by preprocessor
                             stop_words=None,  # stop words can be handled by preprecossor
                             max_features=2000,  # pick top 10 words by frequency
                             # use only unigram counts
                             ngram_range=(1, 1),
                             binary=False)  # we want frequency count features

pipeline = Pipeline([('vec', vectorizer), ('tfidf', TfidfTransformer())])

X = pipeline.fit_transform(data['tokenized'])
Y = data['label'].values
trainX = X[:index]
trainY = Y[:index]
validX = X[index:]
validY = Y[index:]

model = LogisticRegression()

model.fit(trainX, trainY)
train_accuracy = (model.predict(trainX) == trainY).astype(float).mean()
valid_accuracy = (model.predict(validX) == validY).astype(float).mean()

# print(model.predict(validX))
# print(validY)
# print(model.predict(validX) == validY)

metrics.plot_confusion_matrix(model, validX, validY)  # doctest: +SKIP
plt.show()

metrics.plot_roc_curve(model, trainX, trainY)  # doctest: +SKIP
plt.show()

y_pred = model.predict(validX)
# print(y_pred)
# print(validY)
# print(type(y_pred[1]))

print(
    f'precision: {metrics.precision_score(validY, y_pred)}')
print(
    f'recall: {metrics.recall_score(validY, y_pred)}')

print(f'training accuracy: {train_accuracy}')
print(f'validation accuracy: {valid_accuracy}')

# y_pred = [0, 1, 0, 0]
# y_true = [0, 1, 0, 1]
# print(metrics.recall_score(y_true, y_pred))
