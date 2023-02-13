# Imports
import os, torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.out = nn.Linear(embed_size, final_hidden_size)
        self.out = nn.Softmax(final_hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def make_mask(self, src):
        N, src_len = src.shape
        src_mask = torch.tril(
            torch.ones((src_len, src_len))
        ).expand(N, 1, src_len, src_len)
        return src_mask.to(self.device)

    def make_pad_mask(self, src):
        # (N, 1, 1, src_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
        
    def forward(self, src, mask, src_key_padding_mask):
        N, seq_length = src.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        embedding = self.dropout(self.word_embedding(src) + self.position_embedding(positions))
        hidden_states = self.transformer_encoder(embedding, mask, src_key_padding_mask)
        hidden_state = hidden_states[0]
        avg_pool = hidden_state.mean(dim=1)
        logits = self.fc(avg_pool)
        return logits



src_pad_idx = 0
trg_pad_idx = 0
src_vocab_size = 10
trg_vocab_size = 10
device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.tensor([[1,5,6,4,3,9,5,2,0], [1,8,7,3,4,5,6,7,2]]).to(device)
trg = torch.tensor([[1,7,4,3,5,9,2,0], [1,5,6,2,4,7,6,2]]).to(device)
model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device).to(device)
out = model(x, trg[:, :-1])
print(out.shape)


# def list_files(filepath):
#     paths = []
#     for root, _, files in os.walk(filepath):
#         for file in files:
#             paths.append(os.path.join(root, file))
#     return paths

# spam_files = list_files(".\\data\\spam")

