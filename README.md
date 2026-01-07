# 📘 Skip-gram Word2Vec from Scratch (Harry Potter)

## 📌 Project Description

This project implements the **Skip-gram version of Word2Vec from scratch** using text from the *Harry Potter* book series.

The goal is to learn meaningful word embeddings by predicting surrounding context words for a given focal word.

This implementation avoids high-level Word2Vec libraries and instead builds the full pipeline:

- Text preprocessing  
- Vocabulary construction  
- Skip-gram dataset creation  
- Neural network training  
- Embedding extraction and interpretation  

---

## 📂 Dataset

The dataset consists of plain-text files:

HP1.txt, HP2.txt, ..., HP7.txt


Each file contains the text of a Harry Potter book.  
Only a subset of the data is used to keep training efficient and interpretable.

---

## ⚙️ Setup Instructions

### 1. Clone the repository


git clone https://github.com/mlungisimajola323/Word2Vec-Skipgram.git
cd Word2Vec-Skipgram

python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows

pip install -r requirements.txt

### Step 5: Model Architecture

- Input layer:
  - **Linear layer** (when using one-hot word representations)
  - **Embedding layer** (when using word indices)
- Linear hidden layer
- Output layer predicting context word probabilities

The network weights are initialized using **small Gaussian noise (values < 0.1)**.  
A **large learning rate** is used to place the model in the *feature learning regime*, which is desirable when learning word embeddings.

📄 `src/model.py`

---

### Step 6: Training

- Loss function: **Cross-Entropy Loss**
- Optimizer: **Stochastic Gradient Descent (SGD)**
- The model is trained on **skip-gram word–context pairs**
- Each context word is treated as an independent training target

📄 `src/train.py`

---

### Step 7: Inference

An inference function is implemented that:

- Accepts a **one-hot encoded word vector**
- Passes it through the trained network
- Outputs the corresponding **learned word embedding**

This function allows direct inspection and analysis of the embedding space.

📄 `src/inference.py`

