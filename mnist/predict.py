import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from train import CNN

# Path to your drawn digit folder
img_folder = "/Users/adarshkumar/Downloads/MyFlaskApp/mnist/test_image"

# Preprocessing parameters
resize_dim = 28   # MNIST images are 28×28
mean, std = (0.1307,), (0.3081,)

def main():
    # ─── Setup device, model, transform ─────────────────────────────────────────

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)

    checkpoint_path = "mnist_cnn.pth"
    ckpt = torch.load(checkpoint_path, map_location=device)
       # support both pure state_dict or full dict
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((resize_dim, resize_dim)),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: 1.0 - x),        # invert white↔black if needed
        transforms.Normalize(mean, std)
    ])

    # ─── Loop through all images in the folder ─────────────────────────────────
    for fn in os.listdir(img_folder):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(img_folder, fn)
        img = Image.open(img_path).convert("L")
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            pred = output.argmax(dim=1, keepdim=True).item()

        print(f"Image: {fn} → Predicted Digit: {pred}")

if __name__ == "__main__":
    main()

import random
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor, Lambda, Normalize

# use same preprocessing as above
test_transform = Compose([
    Resize((resize_dim, resize_dim)),
    ToTensor(),
    Lambda(lambda x: 1.0 - x),
    Normalize(mean, std)
])

# load MNIST test set
mnist_test = datasets.MNIST(
    '.', train=False, download=True, transform=test_transform
)
indices = random.sample(range(len(mnist_test)), 10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

checkpoint_path = "mnist_cnn.pth"
ckpt = torch.load(checkpoint_path, map_location=device)
    # support both pure state_dict or full dict
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    state_dict = ckpt['model_state_dict']
else:
    state_dict = ckpt
model.load_state_dict(state_dict)
model.eval()

print("\nRandom MNIST test predictions:")
for idx in indices:
    img, label = mnist_test[idx]
    tensor = img.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        pred = out.argmax(dim=1).item()
    print(f"Sample {idx}: Actual={label}, Predicted={pred}")