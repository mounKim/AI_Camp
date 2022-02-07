import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

class MultiLayer(torch.nn.Module):
    def __init__(self):
        super(MultiLayer, self).__init__()
        self.lin_1 = torch.nn.Linear(28*28, 200)
        self.lin_2 = torch.nn.Linear(200, 10)
        torch.nn.init.kaiming_normal_(self.lin_1.weight)
        torch.nn.init.zeros_(self.lin_1.bias)
        torch.nn.init.kaiming_normal_(self.lin_2.weight)
        torch.nn.init.zeros_(self.lin_2.bias)
        
    def forward(self, x):
        net = x
        net = self.lin_1(net)
        net = torch.nn.functional.relu(net)
        net = self.lin_2(net)
        return net

def eval(model, data):
    with torch.no_grad():
        model.eval()
        total, correct = 0, 0
        for x, y in data:
            _, y_pred = torch.max(model(x.view(-1, 28*28)), 1)
            correct += (y_pred == y).sum().item()
            total += x.size(0)
        model.train()
    return correct / total

train = datasets.MNIST(root = "./data/", train = True, transform = transforms.ToTensor(), download = True)
test = datasets.MNIST(root = "./data/", train = False, transform = transforms.ToTensor(), download = True)
train_loader = torch.utils.data.DataLoader(train, batch_size = 256, shuffle = True)
test_loader = torch.utils.data.DataLoader(test, batch_size = 256, shuffle = True)

model = MultiLayer()
loss = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr = 0.001)

x_numpy = np.random.rand(2, 784)
x_torch = torch.from_numpy(x_numpy).float()
y_torch = model(x_torch)
y_numpy = y_torch.detach().cpu().numpy()

'''
np.set_printoptions(precision = 3)
for i, (name, x) in enumerate(model.named_parameters()):
    parameter = x.detach().cpu().numpy()
    print("name:[%s] shape[%s]" % (name, parameter.shape))
    print("val:%s" % parameter.reshape(-1)[:5])
'''

train_accr = eval(model, train_loader)
test_accr = eval(model, test_loader)
print("before train")
print("train_accr:[%.3f] test_accr:[%.3f]" % (train_accr, test_accr))
print("start train")

for epoch in range(10):
    loss_sum = 0
    for x, y in train_loader:
        y_hat = model(x.view(-1, 28*28))
        loss_out = loss(y_hat, y)
        optim.zero_grad()
        loss_out.backward()
        optim.step()
        loss_sum += loss_out
    train_accr = eval(model, train_loader)
    test_accr = eval(model, test_loader)
    print("epoch:[%d] loss:[%.3f] train_accr:[%.3f] test_accr:[%.3f]" % (epoch, loss_sum / len(train_loader), train_accr, test_accr))