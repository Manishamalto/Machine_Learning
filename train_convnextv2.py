import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import timm

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 64
num_epochs = 100
learning_rate = 0.1
momentum = 0.9
weight_decay = 5e-4
num_classes = 100  # CIFAR-100 has 100 classes

# Data augmentation and normalization for training
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std =[0.2675, 0.2565, 0.2761]),
])

# Normalization for validation
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std =[0.2675, 0.2565, 0.2761]),
])

# Datasets and DataLoaders
train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transforms)
val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=val_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Choose model scale: options include 'convnextv2_tiny', 'convnextv2_small', 'convnextv2_base', etc.
model_name = 'convnextv2_tiny'

# Initialize the model
model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training and validation loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_train += torch.sum(preds == labels.data)
        total_train += labels.size(0)

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # Adjust learning rate
    scheduler.step()

    # Calculate average loss and accuracy
    epoch_loss = running_loss / total_train
    epoch_acc = correct_train.double() / total_train

    print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {epoch_loss:.4f}, Training Acc: {epoch_acc:.4f}")

    # Validation phase
    model.eval()
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct_val += torch.sum(preds == labels.data)
            total_val += labels.size(0)

    val_acc = correct_val.double() / total_val
    print(f"Epoch [{epoch+1}/{num_epochs}] Validation Acc: {val_acc:.4f}")

# Save the trained model
torch.save(model.state_dict(), f"{model_name}_cifar100.pth")
print("Training complete and model saved.")