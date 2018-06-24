from textdatasets import ANNBasedTripletSelector, TripletDataset
from losses import TripletLoss
from NetworksForText import EmbeddingNet
from networks import TripletNet
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from trainer import fit
import torch
import numpy as np


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = []
        for text in dataloader:
            if cuda:
                text = text.cuda()
            result = model.get_embedding(text).data.cpu().numpy()
            embeddings.append(result)
    return torch.FloatTensor(embeddings)


batch_size = 40

selector = ANNBasedTripletSelector('../fuzzyjoiner/names_to_cleanse/peoplesNames.txt')
triples_train = TripletDataset(selector, True)
triples_test = TripletDataset(selector, False)

cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = DataLoader(triples_train, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(triples_test, batch_size=batch_size, shuffle=False, **kwargs)

margin = 1.
embedding_net = EmbeddingNet(100, 128, 128)
model = TripletNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = TripletLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 100
fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

new_embeddings = extract_embeddings(selector.get_test_dataset(), model)
selector.get_nn_stats(new_embeddings, False)

