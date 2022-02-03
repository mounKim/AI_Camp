import torch, os
import urllib.request as urllib
import torch.nn as nn
import matplotlib.pyplot as plt
from zipfile import ZipFile
from torchvision import datasets, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loss(yhat, y):
    m = y.size()[1]
    return - (1 / m) * torch.sum(y * torch.log(yhat) + (1 - y) * torch.log(1 - yhat))

def predict(yhat, y):
    y_prediction = torch.zeros(1, y.size()[1])
    for i in range(yhat.size()[1]):
        if yhat[0, i] <= 0.5:
            y_prediction[0, i] = 0
        else:
            y_prediction[0, i] = 1
    return 100 - torch.mean(torch.abs(y_prediction - y)) * 100

def question(x, yhat):
    plt.imshow(x[0].T.numpy())
    plt.show()
    if yhat[0, 0] <= 0.5:
        answer = "ant"
    else:
        answer = "bee"
    print("prediction is " + answer)

class LogisticRegression(nn.Module):
    def __init__(self, dim, lr = torch.scalar_tensor(0.01)):
        super(LogisticRegression, self).__init__()
        self.w = torch.zeros(dim, 1, dtype = torch.float).to(device)
        self.b = torch.scalar_tensor(0).to(device)
        self.grads = {
            "dw": torch.zeros(dim, 1, dtype = torch.float).to(device),
            "db": torch.scalar_tensor(0).to(device)
        }
        self.lr = lr.to(device)

    def forward(self, x):
        z = torch.mm(self.w.T, x) + self.b
        a = self.sigmoid(z)
        return a

    def sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))

    def optimize(self):
        self.w -= self.lr * self.grads["dw"]
        self.b -= self.lr * self.grads["db"]

    def backward(self, x, yhat, y):
        self.grads["dw"] = (1 / x.shape[1]) * torch.mm(x, (yhat - y).T)
        self.grads["db"] = (1 / x.shape[1]) * torch.sum(yhat - y)

if not os.path.exists("./hymenoptera_data"):
    DATA_PATH = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
    urllib.urlretrieve(DATA_PATH, "hymenoptera_data.zip")
    with ZipFile("hymenoptera_data.zip", 'r') as f:
        f.extractall()

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join("./hymenoptera_data", x), data_transforms[x])
                  for x in ['train', 'val']}
train_dataset = torch.utils.data.DataLoader(image_datasets['train'],
                                            batch_size=len(image_datasets['train']),
                                            shuffle=True)
test_dataset = torch.utils.data.DataLoader(image_datasets['val'],
                                           batch_size=len(image_datasets['val']),
                                           shuffle=True)

x, y = next(iter(train_dataset))
loss_list = []
dim = x.reshape(x.shape[0], -1,).T.shape[0]
model = LogisticRegression(dim, torch.scalar_tensor(0.0001).to(device))
model.to(device)

for i in range(11):
    x, y = next(iter(train_dataset))
    test_x, test_y = next(iter(test_dataset))
    x = x.reshape(x.shape[0], -1,).T
    y = y.unsqueeze(0)
    test_x = test_x.reshape(test_x.shape[0], -1,).T
    test_y = test_y.unsqueeze(0)
    yhat = model.forward(x.to(device))
    cost = loss(yhat.data.cpu(), y)
    train_predict = predict(yhat, y)

    model.backward(x.to(device), yhat.to(device), y.to(device))
    model.optimize()

    yhat = model.forward(test_x.to(device))
    test_predict = predict(yhat, test_y)

    if i % 10 == 0:
        loss_list.append(cost)
        print("iter {}: loss {}, train_acc {}, test_acc {}".format(i, cost, train_predict, test_predict))

x, y = next(iter(test_dataset))
'''
plt.plot(loss_list)
plt.show()
'''
question(x, yhat)