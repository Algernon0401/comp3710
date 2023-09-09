import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import gc

# IO Paths
DATA_PATH = './data'
MODEL_PATH = './cifar10_net.pth'

# CNNs
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,24,5) # i, o (width), k
        self.conv2 = nn.Conv2d(24,16,5)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16*5*5, 120) # 16*5*5=400
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DeeperNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3) # i, o (width), k
        self.conv2 = nn.Conv2d(16,24,3)
        self.conv3 = nn.Conv2d(24,16,3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16*2*2, 120) # 16*2*2=64
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# lecture 31 Aug
# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 =

# https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512, num_classes)
        self.fc = nn.Linear(512*1*1, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x) # rm when wrong
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def load_data(data_dir='./data',
                batch_size=128,
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
        dataset = datasets.CIFAR10(
          root=data_dir, train=False,
          download=True, transform=transform,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

        return data_loader
    
    else:
        
        # load the dataset
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=transform,
        )

        valid_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=transform,
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

def train(model, criterion, optimizer,trainloader, validloader, n_epochs, device):
    # f_print = 1
    print("Start Training")
    for epoch in range(n_epochs):
        # running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # move tensors to device
            inputs, labels = data[0].to(device), data[1].to(device)

            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # backward & optimize
            optimizer.zero_grad() # zero the parameter gradients
            loss.backward()
            optimizer.step()

            # clean
            del inputs, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()

            # print statistics
            # running_loss += loss.item()
            # if i % f_print == f_print-1:    # print every f_print mini-batches
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / f_print:.3f}')
            #     running_loss = 0.0
        print ('Epoch [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, n_epochs, loss.item()))
        
         # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in validloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs
    
        print('Accuracy on the {} validation images: {} %'.format(5000, 100 * correct / total)) 

    print("Finished Training")
    return model
        
def test(model, testloader, device):
    # net = torchvision.models.resnet18()
    # net.load_state_dict(torch.load(PATH, map_location=device))
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

def main():
    print("Program start.")
    # Device Config
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning: CUDA not Found. Using CPU.")

    # Hyper-Params
    n_epochs = 8
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

    model = train(model, criterion, optimizer, trainloader, validloader, n_epochs, device)
    torch.save(model.state_dict(), MODEL_PATH)

    accuracy = test(model, testloader, device)
    print('Accuracy on the {} test images: {} %'.format(10000, 100 * accuracy))


if __name__ == "__main__":
    main()

