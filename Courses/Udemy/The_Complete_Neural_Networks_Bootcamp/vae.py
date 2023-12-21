import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image


# Define hyperparameters
image_size = 784
hidden_dim = 400
latent_dim = 20
batch_size = 128
epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='Udemy\The_Complete_Neural_Networks_Bootcamp\data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='Udemy\The_Complete_Neural_Networks_Bootcamp\data',
                                          train=False,
                                          transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=True)

# Create directory to save the reconstructed and sampled images (if directory not present)
sample_dir = r'Udemy\The_Complete_Neural_Networks_Bootcamp\results'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


class VAE(nn.Module):
    def __init__(self) -> None:
        super(VAE, self).__init__()

        # Encoder layers
        self.fc1 = nn.Linear(image_size, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, image_size)

    def encode(self, X):
        h = F.relu(self.fc1(X))
        mu = self.fc2_mean(h)
        log_var = self.fc2_logvar(h)

        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        epsilon = torch.randn_like(std)
    
        return mu + epsilon * std
    
    def decode(self, Z):
        h = F.relu(self.fc3(Z))
        out = torch.sigmoid(self.fc4(h))
        
        return out
    
    def forward(self, X):
        mu, log_var = self.encode(X.view(-1, image_size))   # Flatten X
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)

        return reconstructed, mu, log_var
    

# Define model and optimizer
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        

def loss_function(reconstructed_image, original_image, mu, logvar):
    bce = F.binary_cross_entropy(reconstructed_image, original_image.view(-1, 784), reduction = 'sum')
    kld = 0.5 * torch.sum(logvar.exp() + mu.pow(2) - 1 - logvar)

    return bce + kld


def train(epoch):
    model.train()
    train_loss = 0
    for index, (images, _) in enumerate(train_loader):
        images = images.to(device)
        reconstructed, mu, log_var = model(images)
        loss = loss_function(reconstructed, images, mu, log_var)
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if index % 100 == 0:
            print(f"Train Epoch {epoch}, Batch: [{index}/{len(train_loader)}], Loss: {loss.item()/len(images):.3f}")

    print()
    print(f"Epoch {epoch}, Avg Loss: {train_loss/len(train_loader.dataset):.3f}")
    print()


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(test_loader):
            images = images.to(device)
            reconstructed, mu, logvar = model(images)
            test_loss += loss_function(reconstructed, images, mu, logvar).item()
            if batch_idx == 0:
                comparison = torch.cat([images[:5], reconstructed.view(batch_size, 1, 28, 28)[:5]])
                save_image(comparison.cpu(), r'Udemy\The_Complete_Neural_Networks_Bootcamp\results/reconstruction_' + str(epoch) + '.png', nrow = 5)

    print(f"Avg Test Loss: {test_loss/len(test_loader.dataset):.3f}")
    print()


if __name__ == "__main__":
    # Main function
    for epoch in range(1, epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            # Get rid of the encoder and sample z from the gaussian ditribution and feed it to the decoder to generate samples
            sample = torch.randn(64,20).to(device)
            generated = model.decode(sample).cpu()
            save_image(generated.view(64,1,28,28), r'Udemy\The_Complete_Neural_Networks_Bootcamp\results/sample_' + str(epoch) + '.png')