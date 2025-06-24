import os
import torch
from torchvision import datasets, transforms
from train import CNN

batch_size = 1000

def main():
    # ─── Preprocessing ───────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # ─── Data loader ────────────────────────────────────────────────────────
    test_ds     = datasets.MNIST('.', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ─── Model setup ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CNN().to(device)

    # ─── Load checkpoint (supports both pure state_dict or full dict) ───────
    checkpoint_path = "mnist_cnn.pth"
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # ─── Evaluation ─────────────────────────────────────────────────────────
    correct = 0
    total   = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    acc = correct / total * 100
    print(f"Test Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()