#  Retina Eye Disease Detection System

An AI-powered web application for automated retinal disease classification and severity estimation using deep learning. This project leverages Convolutional Neural Networks (CNNs) to analyze fundus images and provide real-time diagnostic insights, helping in early detection of vision-threatening diseases.

---

## 🚀 Overview

This system detects and classifies retinal conditions into four categories:

- Cataract  
- Diabetic Retinopathy  
- Glaucoma  
- Normal  

Along with classification, it also predicts a severity score (0–1 scale) to indicate the extent of the disease, enabling better clinical interpretation and decision-making.

---

## 💡 Key Features

- 📷 Upload retina (fundus) image for instant analysis  
- 🧠 Deep learning-based multi-output CNN model  
- 📊 Classification + severity prediction  
- ⚡ Real-time results using Streamlit UI  
- 📈 Displays class probabilities and medical insights  

---

## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras – Model development  
- NumPy & PIL – Image preprocessing  
- Streamlit – Interactive web interface  

---

## 🔍 How It Works

- User uploads a retina image  
- Image is preprocessed (resized, normalized)  
- CNN model performs:  
  - Disease classification  
  - Severity estimation  
- Results are displayed with probabilities and severity score  

---

## 🧠 Model Architecture

- CNN-based feature extraction  
- Dense layers for classification (Softmax output)  
- Separate regression head for severity prediction  
- Multi-task learning approach (classification + regression)  

---

## 📊 Evaluation Metrics

- Accuracy, Precision, Recall, F1-score (classification)  
- Mean Squared Error (severity prediction)  
- Confusion Matrix for performance analysis  

---

## 🎯 Use Cases

- Early screening of retinal diseases  
- Assist ophthalmologists in diagnosis  
- Useful in rural/low-resource healthcare settings  
- Educational tool for medical AI applications  

---

## 🔮 Future Improvements

- Integration of Grad-CAM for explainability  
- Larger and more diverse dataset  
- Attention-based architectures for improved accuracy  
- Deployment on cloud for scalability  

---

## 📌 Conclusion

This project demonstrates how AI can enhance healthcare by providing fast, reliable, and accessible retinal disease screening, reducing dependency on manual diagnosis and enabling early intervention.
