Skip-gram Word2Vec from Scratch (Harry Potter)

1. Project Description

  This project implements the Skip-gram version of Word2Vec from scratch using text from the Harry Potter book series.
  The goal is to learn meaningful word embeddings by predicting surrounding context words for a given focal word.
  
  This implementation avoids high-level Word2Vec libraries and instead builds the full pipeline:
  
  - text preprocessing
  
  - dataset creation
  
  - neural network training
  
  - embedding extraction and analysis

2. Dataset
   The dataset consists of plain-text files:
   
   HP1.txt, HP2.txt, ..., HP7.txt
   
   Each file contains the text of a Harry Potter book.

⚙️ Setup Instructions

1. Clone the repository:
   
   git clone https://github.com/mlungisimajola323/Word2Vec-Skipgram.git
   cd Word2Vec-Skipgram

2. Create virtual environment
   
  python -m venv venv
  source venv/bin/activate  # Linux

3. Install dependencies
   
   pip install -r requirements.txt

Project Steps

Step 1: Text Preprocessing

- Read text from a Harry Potter book

- Convert text to lowercase

- Remove punctuation

- Split text by whitespace

 src/preprocess.py

Step 2: Vocabulary Construction

 - Extract unique words

- Preserve order of first appearance

Create:

  - word → index mapping
  
  - index → word mapping

  src/preprocess.py

Step 3: Word Representation

Can take either of these 2 approaches:

  - One-hot encoding
  - Index-based encoding (used with Embedding layer)
    
  src/dataset.py

Step 4: Skip-gram Dataset Creation

  For each word in the text:
  
  - Use the word as input
  - Use its 2-word context window as labels
  - Each context word is treated as a separate training example

  For Example:
  Sentence: "harry went to hogwarts"
  Input: "went"
  Labels: ["harry", "to"]

Step 5: Model Architecture

  - Input layer:
  
    - Linear layer (for one-hot)
    - Embedding layer (for index input)
  
  - Linear hidden layer
  
  - Output layer predicting context word probabilities

  Weights initialized using small Gaussian noise (< 0.1)
  Large learning rate used to encourage feature learning
  
  src/model.py

Step 6: Training

  - Loss: Cross-Entropy
  - Optimizer: SGD
  - Trains on skip-gram pairs
    
  src/train.py

Step 7: Inference

  - An inference function is implemented that:
  - Takes a one-hot vector
  - Outputs the learned embedding

  src/inference.py




  








 

 


   

   

   
   
