# 🎗️ Breast Cancer Support Platform

A Flask-based web application that provides **support, awareness, and tools** for breast cancer detection and education. Leveraging **Artificial Intelligence (AI)** and **Machine Learning (ML)**, the platform includes features like **tumor prediction**, **ultrasound image classification**, **article summarization**, a **multilingual chatbot**, and a **PDF-based question-answering system**.

---

## 🔍 Overview

This project aims to:
- Assist in the **early detection** of breast cancer.
- Provide **educational resources** and tools.
- Enable **interactive support** via AI-powered chat.
- Support **multilingual audiences**: English, French, and Arabic.

---

## 🚀 Features

- **🔐 User Authentication**: Secure login and signup system with hashed passwords and strong validation (12+ characters, including uppercase, lowercase, digits, and special characters).

- **🧪 Tumor Prediction**: Predicts if a tumor is **benign or malignant** using a pre-trained SVM model based on tumor measurements.

- **🖼️ Ultrasound Image Analysis**: Classifies uploaded ultrasound images as **Normal**, **Benign**, or **Malignant** using a CNN model with confidence scores.

- **💬 Multilingual Chatbot**: An AI assistant powered by Groq’s **LLaMA 3.3 70B** model that answers breast cancer-related questions in English, French, or Arabic.

- **📄 PDF Q&A**: Users can upload breast cancer-related PDFs and ask questions about their content using a **Retrieval-Augmented Generation (RAG)** system powered by **FAISS** and **HuggingFace embeddings**.

- **📰 Article Summarizer**: Summarizes web articles or custom input text related to breast cancer using keyword and frequency-based extraction techniques.

- **📢 News & Testimonials**: Displays curated stories and user testimonials for engagement and education.

- **💖 Donation Page**: Placeholder page for future donation features to support breast cancer causes.

- **📱 Responsive UI**: Optimized templates for desktop and mobile devices across all platform pages.

---

## 🧰 Technologies Used

### 🔧 Backend
- Flask
- Python

### 🤖 Machine Learning
- `svm_model_10features.pkl`: SVM for tumor classification
- `scaler_10features.pkl`: Feature scaler
- `backup_model_augmented.h5`: CNN for ultrasound analysis

### 🧠 AI & NLP
- **Groq API** for chatbot (LLaMA 3.3-70B)
- **LangChain** for PDF-based Q&A
- **HuggingFace embeddings** (`all-MiniLM-L6-v2`)
- **FAISS** for fast document retrieval

### 🛠️ Others
- **BeautifulSoup** for web scraping
- **PyPDFLoader** for PDF parsing
- **PIL (Pillow)** for image preprocessing
- **NumPy, Pandas, Joblib, TensorFlow**
- **dotenv** for secure API key management
- **Langdetect** for language detection

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone  https://github.com/RaneemBenMrad/BreastCancerSupportAI.git
cd BreastCancerSupportAI
