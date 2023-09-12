
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
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
INPUT_DIM = 256*256 # dimension of Input
Z_DIM = 20          # dimension of Latent Space
H_DIM = 200         # dimension of Hidden Layer
NUM_EPOCHS = 10     # number of epoch
BATCH_SIZE = 32     # batch size
LR_RATE = 3e-4      # learning rate

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
            # self.target_folder = TRAIN_TARGET_PATH
        elif self.type == DataType.VALID:   # get validating data
            file_annotation = root + VALID_TXT
            self.input_folder = VALID_INPUT_PATH
            # self.target_folder = VALID_TARGET_PATH
        elif self.type == DataType.TEST:    # get testing data
            file_annotation = root + TEST_TXT
            self.input_folder = TEST_INPUT_PATH
            # self.target_folder = TEST_TARGET_PATH

        f = open(file_annotation, 'r') # open file in read only
        data_dict = f.readlines() # get all lines from file
        f.close() # close file

        self.inputs = []
        # self.target_filenames = []
        self.labels = []
        for line in data_dict:
            img_names = line.split() # slipt by ' ', [0]: input, [1]: target
            input = Image.open(self.input_folder + img_names[0])    # read input img
            # target_img = Image.open(self.target_folder + img_names[1])  # read target img
            if self.transform:
                input = self.transform(input)
            # input_img = np.array(input_img.convert('L'))/255    # convert to 8-bit grayscale & normalize
            # target_img = np.array(target_img.convert('L'))/255  # convert to 8-bit grayscale & normalize
            self.inputs.append(input)
            # self.target_filenames.append(img_names[1])
            self.labels.append(img_names[1])
        # self.input_filenames = np.array(self.input_filenames)   # convert to np array
        # self.target_filenames = np.array(self.target_filenames) # convert to np array

    def __getitem__(self, index):
        input = self.inputs[index]
        # target_img_path = self.target_folder + self.target_filenames[index]
        # target_img = Image.open(target_img_path)
        label = int(self.labels[index])
            # target_img = self.transform(target_img)
        return input, label
    
    def __len__(self):
        return len(self.inputs)

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

def train(trainloader, n_epochs, model, optimizer, criterion):
    # Start training
    for epoch in range(n_epochs):
        loop = tqdm(enumerate(trainloader))
        for i, (x, y) in loop:
            # Forward pass
            x = x.to(device).view(-1, INPUT_DIM)
            # y = y.to(device).view(-1, INPUT_DIM)
            x_reconst, mu, sigma = model(x)

            # loss, formulas
            reconst_loss = criterion(x_reconst, x)                                          # Reconstruction Loss
            kl_div = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))    # Kullback-Leibler Divergence

            # Backprop and optimize
            loss = reconst_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
        print ('Epoch [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, n_epochs, loss.item()))

    return model

def inference(model, dataloader, digit, num_examples=1):
    """
    Generates (num_examples) of a particular digit.
    Specifically we extract an example of each digit,
    then after we have the mu, sigma representation for
    each digit we can sample from that.

    After we sample we can run the decoder part of the VAE
    and generate examples.
    """
    print("Generating...")
    for i, data in enumerate(dataloader):
        input, label = data[0][0].to(device), data[1][0].to(device)
        if label == digit:
            image = input

    with torch.no_grad():
        mu, sigma = model.encode(image.view(1, 256*256))

    for example in range(num_examples):
        z = mu + sigma * torch.randn_like(sigma) # mu + sigma + epsilon
        out = model.decode(z).view(-1, 1, 256, 256)
        save_image(out, f"./gened_imgs/generated_{digit}_ex{example}.png")

def load_data(data_dir='./data',
                batch_size=256,
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test=False):
  
    normalize = transforms.Normalize(
        mean=[0.5],
        std=[0.2],
    )

    # define transforms
    transform = transforms.Compose([
            # transforms.Resize((224,224)),
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
    print("Device:", device)

    # Data
    print("Loading Data...")
    trainloader, validloader = load_data(test=False)
    testloader = load_data(batch_size=1,test=True)
    
    # Model, Loss, Optmizer
    model = VariationalAutoEncoder(INPUT_DIM, Z_DIM).to(device)
    criterion = nn.BCELoss(reduction="sum") # loss_func
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)

    # Train
    train_start_time = time.time()
    model = train(trainloader, NUM_EPOCHS, model, optimizer, criterion)
    print("Training Time: %.2f min" % ((time.time() - train_start_time) / 60))
    torch.save(model.state_dict(), MODEL_PATH)

    # Test
    # model.load_state_dict(torch.load(MODEL_PATH))
    # model.eval()
    inference(model, testloader, 1, num_examples=2)
    print("Execution Time: %.2f min" % ((time.time() - start_time) / 60))

def show_img():
    train_dataset = OASIS_MRI(DATA_PATH,type=DataType.TRAIN,transform=transforms.ToTensor())
    # test_dataset = OASIS_MRI(DATA_PATH,type=DataType.TEST,transform=transforms.ToTensor())

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=6, shuffle=True)

    for step ,(b_x,b_y) in enumerate(train_loader):
        if step < 3:
            imgs = torchvision.utils.make_grid(b_y) # b_x: input, b_y: target
            imgs = np.transpose(imgs,(1,2,0))
            plt.imshow(imgs)
            plt.show()

if __name__ == "__main__":
    main()

