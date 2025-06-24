import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# ─── CONFIGURABLE HYPERPARAMETERS ──────────────────────────────────────────
batch_size    = 64        # samples per training batch
epochs        = 20        # total epochs you want to reach (will resume if checkpoint exists)
learning_rate = 0.001     # optimizer step size

# CNN architecture parameters
conv1_out     = 32
conv2_out     = 64
conv3_out     = 128
kernel_size   = 3
dropout_feat  = 0.25
fc1_units     = 256
fc2_units     = 128
dropout_fc    = 0.5
# ───────────────────────────────────────────────────────────────────────────

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, conv1_out, kernel_size, padding=1),
            nn.BatchNorm2d(conv1_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(dropout_feat),

            nn.Conv2d(conv1_out, conv2_out, kernel_size, padding=1),
            nn.BatchNorm2d(conv2_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(dropout_feat),

            nn.Conv2d(conv2_out, conv3_out, kernel_size, padding=1),
            nn.BatchNorm2d(conv3_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(dropout_feat)
        )
        self.classifier = nn.Sequential(
            nn.Linear(conv3_out * 3 * 3, fc1_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(fc2_units, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def main():
    # ─── Data augmentation & preprocessing ───────────────────────────────
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1,0.1), scale=(0.9,1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # ─── Datasets and loaders ────────────────────────────────────────────
    train_ds = datasets.MNIST('.', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ds   = datasets.MNIST('.', train=False, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ─── Setup model, loss, optimizer ───────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # ─── Checkpoint path & resume ────────────────────────────────────────
    checkpoint_path = "mnist_cnn.pth"
    start_epoch = 1
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        # support both full-dict or state_dict only
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt.get('optimizer_state_dict', optimizer.state_dict()))
            start_epoch = ckpt.get('epoch', 0) + 1
        else:
            model.load_state_dict(ckpt)
        print(f"Resuming from epoch {start_epoch}")

    # ─── Training & validation loop ─────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = train_correct = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        model.eval()
        val_loss = val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        # ─── Epoch statistics ────────────────────────────────────────────
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss: {train_loss/len(train_loader):.4f}, "
            f"Acc: {train_correct/len(train_loader.dataset):.4f} | "
            f"Val loss: {val_loss/len(val_loader):.4f}, "
            f"Acc: {val_correct/len(val_loader.dataset):.4f}"
        )

        # ─── Save checkpoint each epoch ───────────────────────────────────
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)

    print(f"Training complete up to epoch {epochs}. Model saved to {checkpoint_path}")

if __name__ == "__main__":
    main()