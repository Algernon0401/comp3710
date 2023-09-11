
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import torchvision.datasets as datasets  
from torch.utils.data import DataLoader, Dataset
import time
import numpy as np
from enum import Enum
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

# IO Paths
DATA_PATH = './data/keras_png_slices_data/'                     # root of data dir
TRAIN_INPUT_PATH = DATA_PATH + 'keras_png_slices_train/'         # train input
VALID_INPUT_PATH = DATA_PATH + 'keras_png_slices_validate/'      # valid input
TEST_INPUT_PATH = DATA_PATH + 'keras_png_slices_test/'           # test input
VALID_TARGET_PATH = DATA_PATH + 'keras_png_slices_seg_validate/' # train target
TRAIN_TARGET_PATH = DATA_PATH + 'keras_png_slices_seg_train/'    # valid target
TEST_TARGET_PATH = DATA_PATH + 'keras_png_slices_seg_test/'      # test target
MODEL_PATH = './p3_vae.pth'         # trained model
TRAIN_TXT = './oasis_train.txt'     # info of img for train
VALID_TXT = './oasis_valid.txt'     # info of img for valid
TEST_TXT = './oasis_test.txt'       # info of img for test

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = 784     # dimension of input
Z_DIM = 20          # dimension of z
H_DIM = 200         # dimension of h
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR_RATE = 3e-4

class DataType(Enum):
    """
        Represents types of datasets
    """
    TRAIN = 1
    VALID = 2
    TEST = 3

# https://sannaperzon.medium.com/paper-summary-variational-autoencoders-with-pytorch-implementation-1b4b23b1763a
# https://zhuanlan.zhihu.com/p/130673468
# https://zhuanlan.zhihu.com/p/163610202
class OASIS_MRI(Dataset):
    def __init__(self, root, type=DataType.TRAIN, transform = None, target_transform=None) -> None:
        super(OASIS_MRI, self).__init__()
        
        self.type = type  # training set / valid set / test set
        self.transform = transform
        self.target_transform = target_transform

        if self.type == DataType.TRAIN:     # get training data
            file_annotation = root + TRAIN_TXT
            self.input_folder = TRAIN_INPUT_PATH
            self.target_folder = TRAIN_TARGET_PATH
        elif self.type == DataType.VALID:   # get validating data
            file_annotation = root + VALID_TXT
            self.input_folder = VALID_INPUT_PATH
            self.target_folder = VALID_TARGET_PATH
        elif self.type == DataType.TEST:    # get testing data
            file_annotation = root + TEST_TXT
            self.input_folder = TEST_INPUT_PATH
            self.target_folder = TEST_TARGET_PATH

        f = open(file_annotation, 'r') # open file in read only
        data_dict = f.readlines() # get all lines from file
        f.close() # close file

        self.input_filenames = []
        self.target_filenames = []
        for line in data_dict:
            img_names = line.split() # slipt by ' ', [0]: input, [1]: target
            # input_img = Image.open(self.input_folder + img_names[0])    # read input img
            # target_img = Image.open(self.target_folder + img_names[1])  # read target img
            # input_img = np.array(input_img.convert('L'))/255    # convert to 8-bit grayscale & normalize
            # target_img = np.array(target_img.convert('L'))/255  # convert to 8-bit grayscale & normalize
            self.input_filenames.append(img_names[0])
            self.target_filenames.append(img_names[1])
        # self.input_filenames = np.array(self.input_filenames)   # convert to np array
        # self.target_filenames = np.array(self.target_filenames) # convert to np array

    def __getitem__(self, index):
        input_img_path = self.input_folder + self.input_filenames[index]
        target_img_path = self.target_folder + self.target_filenames[index]
        input_img = self.transform(Image.open(input_img_path))
        target_img = self.transform(Image.open(target_img_path))
        return input_img, target_img
    
    def __len__(self):
        return len(self.input_filenames)

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

def train(train_loader, num_epochs, model, optimizer, criterion):
    # Start training
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader))
        for i, (x, y) in loop:
            # Forward pass
            x = x.to(device).view(-1, INPUT_DIM)
            x_reconst, mu, sigma = model(x)

            # loss, formulas from https://www.youtube.com/watch?v=igP03FXZqgo&t=2182s
            reconst_loss = criterion(x_reconst, x)
            kl_div = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            # Backprop and optimize
            loss = reconst_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

def test(testloader, model):
    print("Start Testing")
    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            # move tensors to device
            inputs, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del inputs, labels, outputs

        accuracy = correct / total
        return accuracy

def inference(model, dataset, digit, num_examples=1):
    """
    Generates (num_examples) of a particular digit.
    Specifically we extract an example of each digit,
    then after we have the mu, sigma representation for
    each digit we can sample from that.

    After we sample we can run the decoder part of the VAE
    and generate examples.
    """
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, 784))
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"generated_{digit}_ex{example}.png")

def load_data(data_dir='./data',
                batch_size=256,
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test=False):
  
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize,
    ])

    if test:
        test_dataset = OASIS_MRI(
          root=DATA_PATH, type=DataType.TEST,
          transform=transform,
        )

        data_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=shuffle
        )

        return data_loader
    
    else:
        
        # load the dataset
        train_dataset = OASIS_MRI(
            root=DATA_PATH, type=DataType.TRAIN,
            transform=transform,
        )

        valid_dataset = OASIS_MRI(
            root=DATA_PATH, type=DataType.VALID,
            transform=transform,
        )

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler)
    
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler)

        return (train_loader, valid_loader)

def main():
    start_time = time.time()
    print("Program Starts")
    print("Device: ", device)

    # Data
    trainloader, validloader = load_data(test=False)
    testloader = load_data(test=True)
    
    # Model, Loss, Optmizer
    model = VariationalAutoEncoder(INPUT_DIM, Z_DIM).to(device)
    criterion = nn.BCELoss(reduction="sum") # loss_func
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)

    train_start_time = time.time()
    model = train(trainloader, NUM_EPOCHS, model, optimizer, criterion)
    print("Training Time: %.2f min" % ((time.time() - train_start_time) / 60))
    torch.save(model.state_dict(), MODEL_PATH)

    test_start_time = time.time()
    accuracy = test(model, testloader, device)
    print("Testing Time: %.2f min" % ((time.time() - test_start_time) / 60))
    print('Accuracy on the {} test images: {} %'.format(10000, 100 * accuracy))
    print("Execution Time: %.2f min" % ((time.time() - start_time) / 60))

def show_img():
    train_dataset = OASIS_MRI(DATA_PATH,type=DataType.TRAIN,transform=transforms.ToTensor())
    test_dataset = OASIS_MRI(DATA_PATH,type=DataType.TEST,transform=transforms.ToTensor())

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=6, shuffle=True)

    for step ,(b_x,b_y) in enumerate(train_loader):
        if step < 3:
            imgs = torchvision.utils.make_grid(b_y) # b_x: input, b_y: target
            imgs = np.transpose(imgs,(1,2,0))
            plt.imshow(imgs)
            plt.show()

if __name__ == "__main__":
    show_img()

