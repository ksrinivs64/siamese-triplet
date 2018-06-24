import torch.nn as nn

class EmbeddingNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EmbeddingNet, self).__init__()
        self.grunet = nn.Sequential(nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=4, batch_first=True))
        sequence_len = 10
        self.linear = nn.Linear(hidden_size * sequence_len, output_size)

    def forward(self, x):
        output, hn = self.grunet(x)
        t = output.contiguous().view(output.size()[0], -1)
        out1 = self.linear(t)
        return out1

    def get_embedding(self, x):
        return self.forward(x)


