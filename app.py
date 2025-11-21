import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io


device = torch.device("cpu")


IMAGE_SIZE = 224
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


data_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# Definisikan nama kelas (sesuai urutan dari Langkah 3)
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
NUM_CLASSES = len(class_names)

class_descriptions = {
    'akiec': "Ini adalah deskripsi untuk **Actinic Keratoses (akiec)**. Seringkali dianggap sebagai lesi pra-kanker...",
    'bcc': "Ini adalah deskripsi untuk **Basal Cell Carcinoma (bcc)**. Ini adalah jenis kanker kulit yang paling umum...",
    'bkl': "Ini adalah deskripsi untuk **Benign Keratosis-like lesions (bkl)**. Ini adalah lesi jinak (bukan kanker) yang umum terjadi...",
    'df': "Ini adalah deskripsi untuk **Dermatofibroma (df)**. Benjolan kecil yang jinak dan tidak berbahaya...",
    'mel': "Ini adalah deskripsi untuk **Melanoma (mel)**. Ini adalah jenis kanker kulit yang paling serius dan berbahaya...",
    'nv': "Ini adalah deskripsi untuk **Melanocytic Nevi (nv)**. Ini adalah tahi lalat biasa yang umumnya jinak...",
    'vasc': "Ini adalah deskripsi untuk **Vascular lesions (vasc)**. Ini adalah lesi yang berkaitan dengan pembuluh darah..."}

# Buat ulang arsitektur model
model = models.resnet50(pretrained=False) # 'pretrained=False' karena kita akan load bobot kita sendiri
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# load w
PATH = "HAM10000_ResNet50_best.pth"

# Load 
model.load_state_dict(torch.load(PATH, map_location=device))

# Setel model evaluasi
model.eval()
model.to(device)


# Fungsi prediksi

def predict_image(image_bytes):
    """
    Menerima 'bytes' dari gambar, melakukan pre-processing,
    dan mengembalikan prediksi.
    """
    try:
        
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Terapkan transformasi
        image_tensor = data_transforms(image).unsqueeze(0) # Tambah dimensi batch (1)
        
        # Pindahkan ke device
        image_tensor = image_tensor.to(device)
        
        # Lakukan prediksi
        with torch.no_grad(): # Matikan perhitungan gradien
            outputs = model(image_tensor)
            
            # Ubah output mentah (logits) menjadi probabilitas (softmax)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Ambil kelas dengan probabilitas tertinggi
            top_prob, top_idx = torch.max(probabilities, 0)
            
            pred_class = class_names[top_idx.item()]
            confidence = top_prob.item()
            
        return pred_class, confidence, probabilities

    except Exception as e:
        st.error(f"Error saat memproses gambar: {e}")
        return None, None, None

# ui strealit

st.set_page_config(layout="wide", page_title="Klasifikasi Lesi Kulit")

# Judul
st.title("Manawi penyakit kulit (HAM10000)")
st.write("Dibuat menggunakan ResNet50 & PyTorch. Model ini dapat memprediksi 7 jenis lesi kulit.")
st.warning("**PERINGATAN:** Ini adalah proyek edukasi dan **TIDAK** dapat digunakan "
    "untuk diagnosis medis sungguhan. Hasil prediksi bisa jadi salah. "
    "Selalu konsultasikan dengan dokter atau ahli dermatologi profesional "
    "untuk masalah kesehatan apa pun.")

# Petunjuk
st.info("Upload gambar lesi kulit (format .jpg atau .png) untuk melihat prediksi model.")

# File Uploader
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Baca 'bytes' dari file yang di-upload
    image_bytes = uploaded_file.getvalue()
    
    # Tampilkan gambar yang di-upload
    st.image(image_bytes, caption='Gambar yang Di-upload', width=300)
    
    # Buat tombol untuk memulai prediksi
    if st.button('Mulai Prediksi jaya jaya jaya'):
        with st.spinner('Model sedang menganalisis...'):
            
            # Panggil fungsi prediksi
            pred_class, confidence, probabilities = predict_image(image_bytes)
            
            if pred_class:
                st.success("Prediksi Selesai!")
                
                # Tampilkan hasil utama
                st.write(f"###  Prediksi Utama: **{pred_class}**")
                st.write(f"###  Keyakinan: **{confidence*100:.2f}%**")
                with st.expander(f" Apa itu '{pred_class}'? (Klik untuk lihat deskripsi)"):
                    # Ambil deskripsi dari kamus
                    description = class_descriptions.get(pred_class, "Deskripsi tidak ditemukan.")
                    st.markdown(description)
                st.warning("Mohon Maaf apabila ada kesalahan karena model ini waktu di training hanya mendapa score 0.7705(77.05%)."
                " **Kesulitan** saat melatih model ini adalah waktunya sangat lama dan beberapa kali kena limit Runtime(T4) oleh google colab.")
               