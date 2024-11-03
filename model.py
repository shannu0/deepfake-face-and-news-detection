import os
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import warnings
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Suppress warnings
warnings.filterwarnings("ignore")

# Set device
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Initialize MTCNN and InceptionResnetV1 for deepfake detection
mtcnn = MTCNN(select_largest=False, post_process=False, device=DEVICE).to(DEVICE).eval()
model = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=1, device=DEVICE)

# Load the pre-trained model checkpoint
checkpoint_path = "resnetinceptionv1_epoch_32.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
else:
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

# Prediction function for deepfake detection
def predict_deepfake(input_image: Image.Image):
    face = mtcnn(input_image)
    if face is None:
        raise Exception('No face detected')
    
    face = face.unsqueeze(0)  # Add batch dimension
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
    prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy().astype('uint8')
    face = face.to(DEVICE).to(torch.float32) / 255.0
    face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()

    # Grad-CAM explainability
    target_layers = [model.block8.branch1[-1]]  # Adjust as per model structure
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(0)]
    grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)[0, :]

    visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
    face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

    # Model prediction
    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        fake_prediction = round(output.item() * 100, 4)  # Round to 4 decimal places
        real_prediction = round((1 - output.item()) * 100, 4)  # Round to 4 decimal places
        final_result = "Real Image Detected" if real_prediction > 50 else "Deepfake Image Detected"

    return {'real': real_prediction, 'fake': fake_prediction, 'final_result': final_result}, face_with_mask
# Load and preprocess the dataset for news detection
true_df = pd.read_csv('True.csv')
fake_df = pd.read_csv('Fake.csv')
true_df['label'] = 0
fake_df['label'] = 1
dataset = pd.concat([true_df[['text', 'label']], fake_df[['text', 'label']]])
dataset = dataset.sample(frac=1).reset_index(drop=True)

# Text cleaning and preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_data(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

dataset['text'] = dataset['text'].apply(clean_data)

# Train-test split and vectorization
vectorizer = TfidfVectorizer(max_features=50000, lowercase=False, ngram_range=(1, 2))
X = dataset['text']
y = dataset['label']
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
vec_train = vectorizer.fit_transform(train_X)
vec_test = vectorizer.transform(test_X)

# Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(vec_train, train_y)

# Prediction function for fake news detection
def predict_news(text):
    cleaned_text = clean_data(text)
    vec_text = vectorizer.transform([cleaned_text])
    prediction = clf.predict(vec_text)
    return "Fake" if prediction == 1 else "Real"
