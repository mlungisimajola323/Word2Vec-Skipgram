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

```bash
git clone https://github.com/mlungisimajola323/Word2Vec-Skipgram.git
cd Word2Vec-Skipgram

python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows

pip install -r requirements.txt
