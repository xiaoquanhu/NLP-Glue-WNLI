# NLP: different NLP models from linear regression to RNN-GRU and to transformers, experimenting on Glue-WNLI dataset.

By simple command line, you can choose which model you want to experiment on:
Quick demonstration in command line: $ python3 run_bash.py [-model (logistic_regression GRU external_GRU transformers)]

For example, use logistic-regression model: $ python3 run_bash.py logistic_regression  

In this example, you can get the ROC accuracy and confussion_matrix of the experiment results on WNLI dataset.
Note: external_GRU model requires to use glove.6B datasets, size of which is too big to upload here, you can download this dataset and put in the right directory specified in path parameters of python code file. Link: http://nlp.stanford.edu/data/glove.6B.zip 

The very detailed experiment info can be found in report.pdf. Otherwise here is the short version to help you to have a taste:

I experimented with Logistic Regression, RNN-GRU (with and without Glove Embeddings) and pre-trained Transformer models on the sequence classification dataset of GLUE-WNLI. We then compared different model architectures’ performances on the pronouns reference disambiguation task. Finally, we discussed this task’s challenges and some related work in the report.


In general, sequence classification tasks are very sensitive to the words’ position in the sequences, which is the main difference from general classification using Bag of Words vectors (TF-Idf). Specifically, Bag of Words ignores the word position and the context information for the current word, which is crucial to disambiguate pronoun reference in the WNLI dataset. As for Fasttext, because of the window size of Continuous Bag of Words ​(​CBOW) or Skip-gram, it is still difficult for the model to capture the whole semantic meanings between the sentence pairs. This means that it remains difficult for the Fasttext model to learn to disambiguate pronoun references.


Even though RNN models take word positions and sequence context into consideration, their limitation is on the longer sequence, especially for whose sequence size is in the hundreds. As for transformer models, the pre-training process has embedded external language knowledge into the models, which is exactly the key to pronoun disambiguate resolution. As stated above, WNLI task is unambiguous for humans because of humans’ external language knowledge. Considering this characteristic of pre-trained transformer models, our hypothesis is that these models would perform better. In this section we train different models on the WNLI dataset to compare and analyse their performances.


