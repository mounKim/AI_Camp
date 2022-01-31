import numpy as np
import torch

class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

x_values = [i for i in range(10)]
x_train = np.array(x_values, dtype=np.float32).reshape(-1, 1)

y_values = [i*i for i in range(10)]
y_train = np.array(y_values, dtype=np.float32).reshape(-1, 1)

model = LinearRegression(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
# GPU 사용 유무
if torch.cuda.is_available():
    model.cuda()

for epoch in range(100):
    if torch.cuda.is_available():
        inputs = torch.from_numpy(x_train).cuda()
        labels = torch.from_numpy(y_train).cuda()
    else:
        inputs = torch.from_numpy(x_train)
        labels = torch.from_numpy(y_train)
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss))

with torch.no_grad():
    if torch.cuda.is_available():
        predicted = model(torch.from_numpy(x_train).cuda()).cpu().data.numpy()
    else:
        predicted = model(torch.from_numpy(x_train)).data.numpy()
    print(predicted)