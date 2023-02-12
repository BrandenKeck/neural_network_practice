# Imports
import os, torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerClassifier(nn.Module):

    def __init__(self, num_classes,
                        num_encoders=6, 
                        embed_size=512,
                        feedfoward_size=2048,
                        attention_heads=8, 
                        input_vocab_size=10000,
                        max_length = 10000,
                        dropout=0.1):
        super(TransformerClassifier, self).__init__()

        self.dropout = dropout
        self.embed_size = embed_size
        self.num_classes = num_classes
        self.input_embedding = nn.Embedding(input_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(embed_size, attention_heads, feedfoward_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_encoders)
        self.fc = nn.Linear(embed_size, num_classes)
        
    def forward(self, x, mask):
        
        input_embedding = self.input_embedding(input_ids)
        hidden_states = self.transformer_encoder(input_embedding)
        hidden_state = hidden_states[0]
        avg_pool = hidden_state.mean(dim=1)
        logits = self.fc(avg_pool)
        return logits


def list_files(filepath):
    paths = []
    for root, _, files in os.walk(filepath):
        for file in files:
            paths.append(os.path.join(root, file))
    return paths

spam_files = list_files(".\\data\\spam")

