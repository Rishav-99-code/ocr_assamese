import os
import torch
from PIL import Image, ImageEnhance
from torchvision import transforms
from train_fixed import SafeCRNN
from char_map import char_to_idx, idx_to_char
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Correction dictionary for post-processing (copied from interface.py)
correction_dict = {
    'কিিিববাাা': 'কিবা',
    'নহয': 'নহয়',
    'আগবাঢি': 'আগবাঢ়ি',
    'কিবাএটাধুনীয়াদিন': 'কিবা এটা ধুনীয়া দিন।',
    'য‌া‌ি': 'জাতি',
    '।': '<empty>'  
}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model (copied from interface.py main function)
try:
    model = SafeCRNN(img_height=32, nn_classes=len(char_to_idx)).to(device)
    model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit()

def decode_output(output, char_map, idx_to_char):
    """Decode CTC output to text."""
    try:
        _, preds = output.max(2)
        preds = preds.transpose(1, 0).cpu().numpy()
        text = ''
        last_char = None
        for idx in preds[0]:
            if idx == char_map.get('<pad>', -1):
                last_char = None
                continue
            current_char = idx_to_char.get(idx, '<unk>')
            if current_char in ['\u200c', '\u200d', '\n']:
                continue
            if current_char != last_char:
                text += current_char
            last_char = current_char
        return text if text else '<empty>'
    except Exception as e:
        print(f"Decoding error: {str(e)}")
        raise

def preprocess_image(image):
    """Preprocess image for model input."""
    image = ImageEnhance.Contrast(image).enhance(2.0)
    transform = transforms.Compose([
        transforms.Resize((32, 300)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = image.convert('L')
    return transform(image).unsqueeze(0)

def predict_image_from_pil(image_pil, model, device):
    """Predict text from a PIL image object."""
    try:
        input_tensor = preprocess_image(image_pil).to(device)
        with torch.no_grad():
            output = model(input_tensor)
        predicted_text = decode_output(output, char_to_idx, idx_to_char)
        corrected_text = correction_dict.get(predicted_text, predicted_text)
        return corrected_text
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Open the image with PIL for processing
            img = Image.open(filepath).convert('RGB') # Convert to RGB to ensure 3 channels

            # Perform OCR
            predicted_text = predict_image_from_pil(img, model, device)
            os.remove(filepath) # Clean up the uploaded file
            return render_template('index.html', predicted_text=predicted_text)
    return render_template('index.html', predicted_text=None)

if __name__ == '__main__':
    app.run(debug=True) 