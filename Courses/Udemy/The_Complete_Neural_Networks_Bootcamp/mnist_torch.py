import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable


INPUT_SIZE = 784
HIDDEN_SIZE = 400
OUTPUT_SIZE = 10
EPOCHS = 10
BATCH_SIZE = 100
LEARNING_RATE = 0.001


train_dataset = datasets.MNIST(root='./Udemy/The_Complete_neural_Networks_Bootcamp/data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./Udemy/The_Complete_neural_Networks_Bootcamp/data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        out = self.fc1(X)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


if __name__ == "__main__":
    net = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    CUDA = torch.cuda.is_available()
    if CUDA:
        net = net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    correct_train = 0
    total_train = 0
    for epoch in range(EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)
            if CUDA:
                images = images.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            total_train += labels.size(0)
            if CUDA:
                correct_train += (predicted.cpu() == labels.cpu()).sum()
            else:
                correct_train += (predicted == labels).sum()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1:02}/{EPOCHS}], Iteration [{i+1}/{len(train_loader)}], Training Loss: {loss.item():.5f}, Training Accuracy: {100*correct_train/total_train:.5f}%")
    
    print("Training Complete")

    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28*28))
        if CUDA:
            images = images.cuda()

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

        if CUDA:
            correct += (predicted.cpu() == labels.cpu()).sum()
        else:
            correct += (predicted == labels).sum()

    print(f'Test Accuracy: {100*correct/total:.5f}')

