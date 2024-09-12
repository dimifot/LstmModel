import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        #self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        #to pernaw relu,droppout
        self.fc = nn.Linear(hidden_dim, 2)  # Binary classification


    def forward(self, x, hidden=None):
        x = self.embedding(x)
        if hidden is None:
            out, hidden = self.lstm(x)
        else:
            out, hidden = self.lstm(x, hidden)

        out, (hn, cn) = self.lstm(x)

        # Use the cell state (long-term memory) instead of the hidden state
        out = self.fc(cn[-1])

        #out = self.fc(out[:, -1, :])  # Use the output of the last time step # check
        return out, hidden