from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Download the data
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

n_samples, h, w = lfw_people.images.shape # 1288, 50, 37

X = lfw_people.images
n_features = X.shape[1]

Y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0] # 7

print(lfw_people.images.shape, lfw_people.data.shape)
print("\nTotal dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d\n" % n_classes)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :, ]
print("X_train shape:", X_train.shape)
trainloader = torch.utils.data.DataLoader(list(zip(X_train,y_train)), batch_size=4,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(list(zip(X_test,y_test)), batch_size=4,
                                          shuffle=True, num_workers=0)

# cnn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        self.conv2 = nn.Conv2d(32,32,3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*11*7, int(h*w/8)) # 32*11*7 = 2464; h*w=50*37 = 1850
        self.fc2 = nn.Linear(int(h*w/8), n_classes*2) # n_c*2=7*2 = 14
        self.fc3 = nn.Linear(n_classes*2, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Net()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

for epoch in range(16):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print("Finished Training")

# PATH = './lfw_net.pth'
# torch.save(net.state_dict(), PATH)
# net = Net()
# net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on the test images: {100 * correct // total} %')

