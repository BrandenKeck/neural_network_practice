# Imports
import os, re, nltk, torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer

class TransformerClassifier(nn.Module):

    def __init__(self, num_classes,
                        device = "cpu",
                        src_pad_idx = 0,
                        num_encoders=6, 
                        embed_size=512,
                        final_hidden_size = 256,
                        feedfoward_size=2048,
                        attention_heads=8, 
                        input_vocab_size=10000,
                        max_length = 10000,
                        dropout=0.1):
        super(TransformerClassifier, self).__init__()

        self.device = device
        self.embed_size = embed_size
        self.num_classes = num_classes
        self.src_pad_idx = src_pad_idx
        self.word_embedding = nn.Embedding(input_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(embed_size, attention_heads, feedfoward_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_encoders)
        self.linear = nn.Linear(embed_size, final_hidden_size)
        self.out = nn.Softmax(num_classes)
        self.dropout = nn.Dropout(dropout)

    def make_mask(self, src):
        N, src_len = src.shape
        src_mask = torch.tril(
            torch.ones((src_len, src_len))
        ).expand(N, 1, src_len, src_len)
        return src_mask.to(self.device)
        
    def forward(self, src):
        N, seq_length = src.shape
        mask = self.make_mask(src)
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        embedding = self.dropout(self.word_embedding(src) + self.position_embedding(positions))
        hidden_states = self.transformer_encoder(embedding, mask)
        output = self.out(self.linear(hidden_states))
        return output

class SpamClassifier():

    def __init__(self, batch_size=1, delta=1.0, epochs=2000, lr=1e-4):
        self.batch_size = batch_size
        self.delta = delta
        self.epochs = epochs
        self.lr = lr
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = TransformerClassifier(2)#.to(device)

    def train(self, X, Y):

        dataset = TensorDataset(torch.Tensor(X), torch.Tensor(Y))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        ll = nn.HuberLoss(delta=self.delta)
        oo = optim.Adam(self.net.parameters(), lr=self.lr)

        # Train the Neural Network
        for epoch in range(self.epochs):

            running_loss = 0.0
            for i, data in enumerate(loader, 0):
                
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                oo.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = ll(outputs, labels)
                loss.backward()
                oo.step()

                # print statistics
                running_loss += loss.item()

            if epoch % 100 == 99:    # print every 100 epochs
                print(f'[{epoch + 1}] loss: {running_loss / i}')
                running_loss = 0.0

        print('Finished Training')

    # Predict from network
    def predict_network(self, X):
        X = torch.Tensor(X)
        return self.net.forward(X)


def list_files(filepath):
    paths = []
    for root, _, files in os.walk(filepath):
        for file in files:
            paths.append(os.path.join(root, file))
    return paths

def build_email_df(paths, label="spam"):
    n = len(paths)
    labels = n*[label]
    texts = n*[""]
    for idx, path in enumerate(paths):
        try:
            with open(path) as f:
                texts[idx] = re.sub('\r?\n', '', ' '.join(f.readlines())).lower()
        except:
            texts[idx] = np.nan
    return pd.DataFrame(
            data=np.array([texts, labels]).T.tolist(),
            columns=["text", "label"]
        )

import warnings
warnings.filterwarnings('ignore')

vectorizer = TfidfVectorizer()
spam_files = list_files(".\\data\\spam")
spam_df = build_email_df(spam_files, 1)
ham_files = list_files(".\\data\\ham")
ham_df = build_email_df(ham_files, 0)
full_df = pd.concat([spam_df, ham_df])
full_df['text'] = full_df["text"].apply(lambda x: nltk.word_tokenize(x))

full_df['tokens'] = vectorizer.fit_transform(full_df['text'])

len(spam_files)
len(ham_files)

class datastruct():

    def __init__(self):
        self.num_words = 0
        self.word2index = {}
        self.index2word = {}

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.num_words += 1

ds = datastruct()
for idx in range(full_df.shape[0]):
    text = list(full_df.loc[0, "text"])[0]
    for word in text:
        ds.add_word(word)




nltk


model = SpamClassifier()
model.train()