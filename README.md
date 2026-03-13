# 🌱 SeedAnalyst: AI-Driven Seed Verifier

SeedAnalyst is an AI-powered tool to verify the authenticity of seeds. It classifies seeds as **Genuine** or **Counterfeit** and provides visual analysis using color histograms, morphology, surface reflectance, and Grad-CAM. The tool supports **rice, wheat, and corn**, with a Gradio interface for easy upload and analysis.

---

## Features

- ✅ AI-based seed classification: Genuine / Counterfeit  
- 📊 Visual analysis:  
  - Color Histogram  
  - Morphology (Area/Perimeter Ratio)  
  - Surface Reflectance  
  - Grad-CAM heatmap for visual explanation  
- 🌾 Crop sanity check for rice, wheat, and corn  
- ⚠️ Warning if uploaded seed may not match the selected crop type  
- 🖥️ Interactive web interface using Gradio  

---

## Installation
OPTION 1:

1.Clone this repository:
git clone https://github.com/pavani0602/SeedAnalyst.git
cd SeedAnalyst

2.Install dependencies:
pip install -r requirements.txt

3.Run the app:
python seed_verifier_app.py

4.Open the Gradio link in your browser to start analyzing seeds.

---

OPTION 2:

You can run SeedAnalyst in Google Colab:

CODE TO BE RUN:

!git clone https://github.com/pavani0602/SeedAnalyst.git

%cd SeedAnalyst

!pip install -r requirements.txt

!python seed_verifier_app.py

A Gradio link will appear — open it to use the app online.

---

Model

File: seed_verifier.h5

Keras-based deep learning model for seed authenticity classification

Size: ~85 MB

Predicts Genuine (🟢) or Counterfeit (🔴) seeds

---



