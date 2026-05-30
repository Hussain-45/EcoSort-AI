import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt

# Configuration Settings
DATA_DIR = "RealWaste"
MODEL_SAVE_PATH = "best_waste_model.pth"
PLOT_SAVE_PATH = "training_curves.png"
BATCH_SIZE = 32
EPOCHS = 5  # Set to a small number for testing; increase (e.g., 10-15) for high accuracy
LEARNING_RATE = 0.0001
VAL_SPLIT = 0.2

# Custom dataset wrapper to apply transforms separately to train/val splits
class SubsetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        # Retrieve the image path and label from the base ImageFolder dataset
        img_path, label = self.subset.dataset.samples[self.subset.indices[index]]
        # Load the PIL Image using the base dataset's default loader
        img = self.subset.dataset.loader(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label

    def __len__(self):
        return len(self.subset.indices)

def train_model():
    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for training: {device}")

    # 2. Data Transformations (Augmentations for training, simple normalizing for validation)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Load entire dataset (without default transform, applied by SubsetWrapper instead)
    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory '{DATA_DIR}' does not exist.")
        return

    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=None)
    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes
    print(f"Loaded {len(full_dataset)} images belonging to {num_classes} classes:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")

    # 4. Train / Val Split
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    raw_train_subset, raw_val_subset = random_split(full_dataset, [train_size, val_size])

    # Wrap subsets with independent transformations
    train_dataset = SubsetWrapper(raw_train_subset, train_transform)
    val_dataset = SubsetWrapper(raw_val_subset, val_transform)

    # 5. Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train split size: {len(train_dataset)} | Validation split size: {len(val_dataset)}")

    # 6. Initialize Model (ResNet18 transfer learning)
    print("Loading pre-trained ResNet18 model...")
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Freeze standard features (optional, but fine-tuning everything yields better accuracy for garbage)
    # For this project, we train all weights with a low learning rate (0.0001)
    
    # Modify final layer to match our class count
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # 7. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 8. Training loop
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("\nStarting Training Loop...")
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = running_corrects.double() / len(train_dataset)
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        running_val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                running_val_loss += loss.item() * inputs.size(0)
                running_val_corrects += torch.sum(preds == labels.data)

        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_acc = running_val_corrects.double() / len(val_dataset)
        
        latency = time.time() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} ({latency:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        
        # Save metrics history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc.item())
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc.item())

        # Checkpoint: Save best model weights
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  --> Saved new best checkpoint with accuracy: {best_acc:.4f}")

    print("\nTraining completed!")
    print(f"Best Validation Accuracy achieved: {best_acc:.4f}")

    # 9. Plotting metrics
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', color='blue')
    plt.plot(history['val_acc'], label='Val Acc', color='red')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH)
    print(f"Saved metric curves to: {PLOT_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
