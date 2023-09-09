
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import torchvision.datasets as datasets  
from torch.utils.data import DataLoader
import time

# IO Paths
DATA_PATH = './data'
MODEL_PATH = './p3_vae.pth'

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = 784
Z_DIM = 20
H_DIM = 200
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR_RATE = 3e-4

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, z_dim, h_dim=200):
        super().__init__()
        # encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)

        # one for mu and one for stds, note how we only output
        # diagonal values of covariance matrix. Here we assume
        # the pixels are conditionally independent 
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        # decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.img_2hid(x))
        mu = self.hid_2mu(h)
        sigma = self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z):
        new_h = F.relu(self.z_2hid(z))
        x = torch.sigmoid(self.hid_2img(new_h))
        return x

    def forward(self, x):
        mu, sigma = self.encode(x)

        # Sample from latent distribution from encoder
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma*epsilon

        x = self.decode(z_reparametrized)
        return x, mu, sigma

def train(train_loader, num_epochs, model, optimizer, loss_fn):
    # Start training
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader))
        for i, (x, y) in loop:
            # Forward pass
            x = x.to(device).view(-1, INPUT_DIM)
            x_reconst, mu, sigma = model(x)

            # loss, formulas from https://www.youtube.com/watch?v=igP03FXZqgo&t=2182s
            reconst_loss = loss_fn(x_reconst, x)
            kl_div = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            # Backprop and optimize
            loss = reconst_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

def main():
    start_time = time.time()
    print("Program Starts")
    # Device Config
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning: CUDA not Found. Using CPU.")

    # Hyper-Params
    n_epochs = 20
    learning_rate = 0.01
    # batch_size = 16
    # n_classes = 10

    # Data
    trainloader, validloader = load_data(test=False)
    testloader = load_data(test=True)
    # total_step = len(trainloader)
    
    # Model, Loss, Optmizer
    model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)
    model.to(device)
    criterion = nn.CrossEntropyLoss() # loss_func
    # optimizer = torch.optim.Adam(net.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)

    train_start_time = time.time()
    model = train(model, criterion, optimizer, trainloader, validloader, n_epochs, device)
    print("Training Time: %.2f min" % ((time.time() - train_start_time) / 60))
    torch.save(model.state_dict(), MODEL_PATH)

    test_start_time = time.time()
    accuracy = test(model, testloader, device)
    print("Testing Time: %.2f min" % ((time.time() - test_start_time) / 60))
    print('Accuracy on the {} test images: {} %'.format(10000, 100 * accuracy))
    print("Execution Time: %.2f min" % ((time.time() - start_time) / 60))


if __name__ == "__main__":
    main()

