# region Import Libraries
import torch
import  datetime
import torchvision
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from matplotlib import pyplot as plt
# endregion


# region Fashion MNIST Data Load
trans_img = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.FashionMNIST("./data/", train=True, transform=trans_img, download=True)
test_dataset = torchvision.datasets.FashionMNIST("./data/", train=False, transform=trans_img, download=True)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle = True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle = True)
# endregion

# Convolution Neural Network Class
class CNN(nn.Module):
    def __init__(self, n_classes=10):
        emb_dim = 512
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, emb_dim)
        self.fc2 = nn.Linear(emb_dim, 128)
        self.clf = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.clf(x)
        return out


def train_CNN(num_epocs):
    device = torch.device("cpu")
    model = CNN(10).to(device)
    loss_list = []
    optimizer = optim.Adam(model.parameters())  # , lr=0.001
    for i in tqdm(range(num_epocs)):
        model.train()
        avg_loss = 0
        count = 0
        train_total = 0
        train_correct = 0
        for batch_idx, (img, target) in enumerate(trainloader):
            img = Variable(img).to(device)
            target = Variable(target).to(device)
            # Zero out the gradients
            optimizer.zero_grad()
            # Forward Propagation
            out = model(img)
            loss = F.cross_entropy(out, target)
            avg_loss = avg_loss + loss
            # backward propagation
            loss.backward()
            # Update the model parameters
            optimizer.step()
            pred = torch.max(out.data, 1)
            train_correct += (pred[1] == target).sum().item()
            count = count + 1
            train_total = train_total + target.size(0)
        loss_list.append(avg_loss / count)
        train_acc = train_correct / train_total
        test_total = 0
        test_correct = 0
        for batch_idx, (img, target) in enumerate(testloader):
            img = Variable(img).to(device)
            target = Variable(target).to(device)
            out = model(img)
            pred = torch.max(out.data, 1)
            test_correct += (pred[1] == target).sum().item()
            test_total = test_total + target.size(0)
        test_acc = test_correct / test_total
        print("Train Loss :" + str((avg_loss / count).data) + ", Train Accuracy :" + str(
            train_acc) + ", Test Accuracy :" + str(test_acc))
    return loss_list

loss_list=train_CNN(25)
plt.figure()
plt.plot(loss_list)
plt.title("training-loss-CNN")
plt.savefig("training_loss_cnn_"+str(datetime.datetime.now())+".jpg")