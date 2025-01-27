import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("Agg")

# Define a custom dataset to load images
class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))]

        if not self.image_files:
            raise ValueError(f"No valid image files found in {image_folder}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img

# Define the Generator model
class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 3 * 64 * 64),
            nn.Tanh()
        )

    def forward(self, z):
        return self.fc(z).view(-1, 3, 64, 64)

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        x = self.conv(img)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Initialize the models, loss function, and optimizers
def initialize_models(z_dim):
    generator = Generator(z_dim)
    discriminator = Discriminator()
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    return generator, discriminator, criterion, optimizer_g, optimizer_d

# Apply weights initialization
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight, mean=1.0, std=0.02)
        nn.init.constant_(m.bias, 0)

# Train the GAN with early stopping and checkpointing
def train_gan(generator, discriminator, dataloader, criterion, optimizer_g, optimizer_d, device, epochs, z_dim, patience=10, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    real_label = 1
    fake_label = 0

    d_losses = []
    g_losses = []
    best_g_loss = float('inf')
    epochs_without_improvement = 0

    ema_generator = Generator(z_dim).to(device)
    ema_generator.load_state_dict(generator.state_dict())
    ema_decay = 0.999

    for epoch in range(epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0
        for imgs in dataloader:
            imgs = imgs.to(device)
            batch_size = imgs.size(0)
            valid = torch.ones(batch_size, 1).to(device) * real_label
            fake = torch.zeros(batch_size, 1).to(device) * fake_label

            # Train the discriminator
            optimizer_d.zero_grad()
            z = torch.randn(batch_size, z_dim).to(device)
            gen_imgs = generator(z)

            real_loss = criterion(discriminator(imgs), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_d.step()
            epoch_d_loss += d_loss.item()

            # Train the generator
            optimizer_g.zero_grad()
            g_loss = -torch.mean(discriminator(gen_imgs))
            g_loss.backward()
            optimizer_g.step()
            epoch_g_loss += g_loss.item()

        epoch_d_loss /= len(dataloader)
        epoch_g_loss /= len(dataloader)
        d_losses.append(epoch_d_loss)
        g_losses.append(epoch_g_loss)

        print(f"[Epoch {epoch + 1}/{epochs}] [D loss: {epoch_d_loss:.4f}] [G loss: {epoch_g_loss:.4f}]")

        # Exponential Moving Average (EMA) update
        for param_ema, param in zip(ema_generator.parameters(), generator.parameters()):
            param_ema.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

        # Check for early stopping
        if epoch_g_loss < best_g_loss:
            best_g_loss = epoch_g_loss
            epochs_without_improvement = 0
            torch.save(generator.state_dict(), os.path.join(checkpoint_dir, "best_generator.pth"))
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

        # Save images after each epoch
        if (epoch + 1) % 10 == 0:
            save_image(gen_imgs.data[:2], f"generated_images_epoch_{epoch + 1}.png", nrow=2, normalize=True)

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("GAN Training Losses")
    plt.savefig("training_losses.png")
    plt.close()

# Generate and save images to a folder
def generate_and_save_images(generator, z_dim, device, num_images=10, save_dir="generated_images"):
    os.makedirs(save_dir, exist_ok=True)
    generator.eval()
    z = torch.randn(num_images, z_dim).to(device)

    with torch.no_grad():
        generated_images = generator(z)

    generated_images = (generated_images + 1) / 2  # Rescale to [0, 1]

    for i in range(num_images):
        image_path = os.path.join(save_dir, f"generated_image_{i + 1}.png")
        save_image(generated_images[i], image_path, normalize=True)
        print(f"Saved generated image to: {image_path}")

# Main function
def main():
    image_folder = "C:\\Users\\Tejaswini Rout\\Downloads\\EXECUTION CODE\\DATA_AUG_IMAGES"
    z_dim = 100
    batch_size = 32
    epochs = 100

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageDataset(image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator, discriminator, criterion, optimizer_g, optimizer_d = initialize_models(z_dim)
    generator.to(device)
    discriminator.to(device)

    train_gan(generator, discriminator, dataloader, criterion, optimizer_g, optimizer_d, device, epochs, z_dim)

    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, "C:\\Users\\Tejaswini Rout\\Downloads\\EXECUTION CODE\\FORGED_IMAGES")
    generate_and_save_images(generator, z_dim=z_dim, device=device, num_images=10, save_dir=save_dir)

if __name__ == "__main__":
    main()
