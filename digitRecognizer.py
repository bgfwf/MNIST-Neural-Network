import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn
import torch

# GET DATA
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,)),])
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
trainsetLoader = torch.utils.data.DataLoader(mnist_trainset, batch_size=64, shuffle=True)
testsetLoader = torch.utils.data.DataLoader(mnist_testset, batch_size=64, shuffle=True)

# Create Model and cost function
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

cost = nn.CrossEntropyLoss();
optimizer = torch.optim.SGD(model.parameters(), lr=.001)

# Training
epochs = 5
for e in range(epochs):
    totalLoss = 0
    for images, labels in trainsetLoader:
        # process image
        images = images.view(images.shape[0], -1)
        
        # reset gradient
        optimizer.zero_grad()
        
        # forward pass
        output = model(images)
            
        # backpropagation
        loss = cost(output, labels)
        loss.backward()
        optimizer.step()
        
        totalLoss += loss.item()
    
    print(f'Average Loss = {totalLoss/len(trainsetLoader)}')
        
# Test set evaluation
testLoss = 0
for images, labels in trainsetLoader:
    images = images.view(images.shape[0], -1)

    output = model(images)
    testLoss = cost(output, labels)
    
print(f'Average Test Loss = {1-loss/len(testsetLoader)}')
