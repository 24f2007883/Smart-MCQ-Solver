# 🧠 Smart MCQ Solver (Scratch Model)

A Multiple Choice Question Answering (MCQ) system built completely from scratch using **Word2Vec** and **PyTorch**. The model predicts the most relevant answer option from five given choices.

---

## 🚀 Features

- Built completely from scratch (No Transformer Models)
- Word2Vec based semantic embeddings
- Custom PyTorch Neural Network
- Predicts Top-3 Answer Options
- Gradio Web Interface
- Hugging Face Ready

---

## 📂 Project Structure

```
smart-mcq-solver/
│── app.py
│── model.py
│── utils.py
│── best_model.pth
│── word2vec.model
│── requirements.txt
│── README.md
```

---

## 🏗 Model Architecture

### Word Embeddings
- Word2Vec
- Vector Size: 100
- Skip-Gram (`sg=1`)
- Window Size: 5

### Neural Network

Input:

```
Prompt Vector
+
Option Vector
+
|Prompt − Option|
```

↓

```
Linear
↓
ReLU
↓
Dropout
↓
Linear
↓
ReLU
↓
Dropout
↓
Linear
```

↓

Compatibility Score

The model computes a score for each of the five answer options and selects the option with the highest score.

---

## 📊 Training Details

- Framework: PyTorch
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Epochs: 30
- Hidden Dimension: 128

---

## 🎯 Prediction Pipeline

```
Question
        │
        ▼
Text Cleaning
        │
        ▼
Word2Vec Embedding
        │
        ▼
Sentence Vector (Mean Pooling)
        │
        ▼
MultipleChoiceModel
        │
        ▼
Softmax
        │
        ▼
Top-3 Predictions
```

---

## ▶️ Run Locally

Clone the repository

```bash
git clone <YOUR_GITHUB_REPO>
cd smart-mcq-solver
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:7860
```

---

## 🛠 Technologies Used

- Python
- PyTorch
- Gensim
- Word2Vec
- NumPy
- Gradio

---

## 📌 Limitations

- Since the model uses Word2Vec, words that were not present during training are treated as Out-of-Vocabulary (OOV). Such words do not have learned embeddings, which may reduce prediction quality on completely unseen text.

---

## 👨‍💻 Author

**Little San**

Built as part of an NLP Multiple Choice Question Answering project.