import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import random


INPUT_SIZE = 784
HIDDEN_SIZE = 400
OUTPUT_SIZE = 10
EPOCHS = 10
BATCH_SIZE = 100
LEARNING_RATE = 0.01


mean_gray = 0.1307
std_gray = 0.3081

transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean_gray,), (std_gray))])

train_dataset = datasets.MNIST(root='./Udemy/The_Complete_neural_Networks_Bootcamp/data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./Udemy/The_Complete_neural_Networks_Bootcamp/data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        #same padding = (filter_size - 1) / 2
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        #(input_size - filter_size + 2(padding)) / stride + 1 = (28 - 3 + 2) / 1 + 1 = 28
        self.batch_norm1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(1568, 600)
        self.drop_out = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(600, 10)

    def forward(self, X):
        out = self.cnn1(X)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.max_pool(out)
        out = self.cnn2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.max_pool(out)
        out = out.view(-1, 1568)    # flatten

        out = self.fc1(out)
        out = self.drop_out(out)
        out = self.fc2(out)

        return out


if __name__ == "__main__":
    model = CNN()
    CUDA = torch.cuda.is_available()
    if CUDA:
        model = model.cuda()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []

    for epoch in range(1, EPOCHS+1):
        correct = 0
        iterations = 0
        iter_loss = 0.0

        model.train()   # activates training mode (treats dropout differently)

        for i, (inputs, labels) in enumerate(train_loader):
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            iter_loss += loss.item()        # loss is tensor

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            iterations += 1
        
        train_loss.append(iter_loss / iterations)
        train_accuracy.append(correct / len(train_dataset))

        # Testing phase
        iter_loss = 0.0
        correct = 0
        iterations = 0

        model.eval()        # activates testing mode (treats dropout differently)

        for i, (inputs, labels) in enumerate(test_loader):
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            iter_loss += loss.item()        # loss is tensor

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            iterations += 1
        
        test_loss.append(iter_loss / iterations)
        test_accuracy.append(correct / len(test_dataset))

        print(f"Epoch [{epoch}/{EPOCHS}], Training Loss: {train_loss[-1]:.5f}, Training Accuracy: {train_accuracy[-1]:.5f}, Test Loss: {test_loss[-1]:.5f}, Test Accuracy: {test_accuracy[-1]:.5f}")

    # Plot results
    # Loss
    fig = plt.figure(figsize=(10,10))
    plt.plot(train_loss, label = "Training Loss")
    plt.plot(test_loss, label = "Test Loss")
    plt.legend()
    plt.show()

    # Loss
    fig = plt.figure(figsize=(10,10))
    plt.plot(train_accuracy, label = "Training Accuracy")
    plt.plot(test_accuracy, label = "Test Accuracy")
    plt.legend()
    plt.show()

    # predict single image
    num = random.randint(0, len(test_dataset))
    img = test_dataset[num][0].resize((1, 1, 28, 28))
    label = test_dataset[num][1]

    model.eval()

    if CUDA:
        model = model.cuda()
        img = img.cuda()

    outputs = model(img)
    _, predicted = torch.max(outputs, 1)
    print(f"Prediction is {predicted.item()}")

    
