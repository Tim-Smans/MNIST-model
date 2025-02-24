from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import io

app = Flask(__name__)

# Define the same model as i used in the notebook
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.fc(x)

# Loading the trained model, this is the file i saved in my notebook. Also setting the model to evaluation mode.
model = MNISTModel()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval() 

# Define the image preprocessing
def preprocess_image(image):
    # Convert the image to grayscale
    image = image.convert("L")
    
    # Invert colors
    image = ImageOps.invert(image)
    
    # Resize and center the image
    image = ImageOps.fit(image, (28, 28), centering=(0.5, 0.5))
    
    # Convert back to black digits on white background
    image = ImageOps.invert(image)
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image_tensor = transform(image).unsqueeze(0) 
    return image_tensor

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Read the image file
        image = Image.open(io.BytesIO(file.read()))
        
        # Preprocess the image
        image_tensor = preprocess_image(image)
        
        # Make a prediction
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            prediction = predicted.item()
        
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
