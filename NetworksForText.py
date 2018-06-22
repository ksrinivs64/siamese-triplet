import torch.nn as nn

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.grunet = nn.Sequential(nn.GRU(hidden_size=128, num_layers=4))

    def forward(self, x):
        output = self.grunet(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)

