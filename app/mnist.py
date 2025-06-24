from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from mnist.train import CNN

main = Blueprint('main', __name__)

# ─── Load your trained model ───────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
# adjust path if needed:
model_path = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "mnist", "mnist_cnn.pth")
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ─── Same transforms as in train.py/predict.py ────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),                      # ensure single‐channel
    transforms.ToTensor(),                       
    transforms.Lambda(lambda x: 1.0 - x),        # <— invert white↔black
    transforms.Normalize((0.1307,), (0.3081,))
])

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        # 1) receive upload
        file = request.files.get('image')
        if not file or file.filename == '':
            flash('Please choose an image file')
            return redirect(url_for('main.profile'))

        # 2) save to uploads folder
        filename = secure_filename(file.filename)
        upload_dir = current_app.config.setdefault(
            'UPLOAD_FOLDER',
            os.path.join(os.path.dirname(__file__), 'static', 'uploads')
        )
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)

        # 3) preprocess & run inference
        img = Image.open(filepath)
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            digit = int(output.argmax(dim=1).item())

        # 4) render result
        return render_template(
            'predict.html',
            name=current_user.name,
            digit=digit,
            image_url=url_for('static', filename='uploads/' + filename)
        )

    # GET
    return render_template('predict.html', name=current_user.name)