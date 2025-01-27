import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Ensure matplotlib works in non-interactive environments
from tqdm import tqdm 

# Set the paths to the folders containing images
REAL_IMAGE_PATH = 'C:\\Users\\Tejaswini Rout\\Downloads\\EXECUTION CODE\\REAL_IMAGES'
FORGED_IMAGE_PATH = 'C:\\Users\\Tejaswini Rout\\Downloads\\EXECUTION CODE\\FORGED_IMAGES'

# Ensure the folders exist
if not os.path.exists(REAL_IMAGE_PATH) or not os.path.exists(FORGED_IMAGE_PATH):
    print("One or both image folders do not exist. Please check the paths.")

# Dataset transformations (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.Lambda(lambda img: img.convert('RGB')),  # Convert to RGB if image is grayscale
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard ImageNet normalization
])

class CustomImageDataset(data.Dataset):
    """
    A custom dataset to load images from two separate folders: real and forged images.
    """
    def __init__(self, real_folder, forged_folder, transform=None):
        self.real_folder = real_folder
        self.forged_folder = forged_folder
        self.transform = transform

        # Load image paths
        self.real_images = [os.path.join(real_folder, f) for f in os.listdir(real_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        self.forged_images = [os.path.join(forged_folder, f) for f in os.listdir(forged_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

        # Labels: 0 for real, 1 for forged
        self.image_paths = self.real_images + self.forged_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.forged_images)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path)  # Open image

        if self.transform:
            image = self.transform(image)  # Apply transformations

        return image, label

# Initialize dataset and split into train and test sets
dataset = CustomImageDataset(REAL_IMAGE_PATH, FORGED_IMAGE_PATH, transform=transform)
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=32, shuffle=False)

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) model for binary classification (real vs. forged).
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass input through convolutional layers and max pooling
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)

        # Flatten the output and pass through fully connected layers
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))

        return x

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.BCELoss()  # Binary Cross-Entropy loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

def plot_performance(history):
    """Plot training/validation loss and accuracy."""
    epochs = range(1, len(history['train_loss']) + 1)

    # Create subplots for Loss and Accuracy
    plt.figure(figsize=(14, 6))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'r-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'b-', label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'r-', label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], 'b-', label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig('performance_plot.png')  # Save plot to file
    plt.show()

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5):
    """
    Train the CNN model and track training and validation performance.
    """
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss, correct, total = 0.0, 0, 0

        # Training phase
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # Calculate training loss and accuracy
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total * 100
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels.float()).item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        # Calculate validation loss and accuracy
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total * 100
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Display results in a table
        headers = ["Epoch", "Train Loss", "Val Loss", "Train Acc (%)", "Val Acc (%)"]
        results = [[epoch + 1, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{train_acc:.2f}", f"{val_acc:.2f}"]]
        print(tabulate(results, headers=headers, tablefmt="grid"))

    # Plot performance after training
    plot_performance(history)
    return history

def inference(model, data_loader):
    """Perform inference on the validation set and print predictions."""
    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()

            # Print predicted vs actual labels
            for i in range(len(predicted)):
                pred = "Forged" if predicted[i].item() == 1 else "Real"
                actual = "Forged" if labels[i].item() == 1 else "Real"
                print(f"Predicted: {pred}, Actual: {actual}")

# Train the model
history = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=2)

# Perform inference on validation set
print("\nSignature Classification:")
inference(model, val_loader)

# Save the trained model
torch.save(model.state_dict(), "image_classifier.pth")
