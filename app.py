import streamlit as st
from PIL import Image
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from model import ConvNet

class ConvNet(nn.Module):
    def __init__(self, dropout_rate = 0.5):
        super(ConvNet, self).__init__()
        self.resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        
        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 8)
            # this will produce the logits; only use activation functions in hidden layers
        )
        
    def forward(self, x):
        return self.resnet(x)

model = ConvNet()

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

checkpoint = torch.load("best_bloodmnist_model.pth", map_location = device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]) # standard for medmnist images

labels = [
    'Basophil',
    'Eosinophil',
    'Erythroblast',
    'Immature Granulocytes',
    'Lymphocyte',
    'Monocyte', 
    'Neutrophil',
    'Platelet'
]

st.title("hema-net classifier 🧬")

uploaded_file = st.file_uploader("Upload a cell image", type = ["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image,caption = 'Uploaded image', use_container_width = True)

    if st.button('Classify'):
        img_tensor = data_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1).squeeze()
            
            confidence, predicted_idx = torch.max(probs, dim = 0)
            predicted_label = labels[predicted_idx.item()]
            st.success(f"Prediction: **{predicted_label}** ({confidence.item():.2%} confidence)")
            
            st.subheader("Confidence for all classes:")
            prob_dict = {label: float(prob) for label, prob in zip(labels, probs)}
            st.bar_chart(prob_dict)