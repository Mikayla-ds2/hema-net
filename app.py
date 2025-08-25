import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

model = torch.load('best_bloodmnist_model.pth', map_location = torch.device('mps'))
model.eval()

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])]) # standard for medmnist images

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
    st.image(image,caption = 'Uploaded image', use_column_width = True)

    if st.button('Classify'):
        img_tensor = data_transform(image).unsqueeze(0)
        with torch.no_grad():
           output =  model(img_tensor)
           _, predicted = torch.max(output, 1)
           st.success(f"Prediction: {labels[predicted.item()]}")