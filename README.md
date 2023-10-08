# Sentimantal_Analysis
# Bully Detection Final Project

## Overview
This repository contains code and data for a Bully Detection project. 
The goal of this project is to identify and clean comments or text data to make it suitable for analysis, particularly focusing on detecting and handling bullying or inappropriate language.
This repository contains code and data for a text classification project aimed at detecting bullying and classifying text comments into different sentiment categories (Negative, Neutral, Positive). 
The project leverages Natural Language Processing (NLP) techniques, machine learning, and the BERT model for text classification.
This repository contains code for building and comparing different Natural Language Processing (NLP) models for text classification. The code includes implementations of the following models:
- BERT (Bidirectional Encoder Representations from Transformers)
- GRU (Gated Recurrent Unit)
- LSTM (Long Short-Term Memory)

These models are trained and evaluated on a text classification task using a dataset that contains comments labeled with different sentiment categories (e.g., Negative, Neutral, Positive).

## Dataset
The project uses the 'BullyDetectionFinal.csv' dataset, which contains comments and related information. You can find the dataset in the `/data` directory.

## Data Cleaning
We've applied several cleaning steps to the dataset to prepare it for analysis. The following steps have been taken:
- Removal of unnecessary columns ('Name', 'Time', 'Likes', 'Reply Count').
- Handling missing values.
- Cleaning of comments to remove emojis, HTML tags, and non-English words.
- Removing non-alphabetical characters.

## Preprocessing
To clean the text data, we've used Python libraries like `clean-text`, `nltk`, and regular expressions. The code for data cleaning and preprocessing can be found in the Jupyter Notebook file `data_preprocessing.ipynb`.

## Usage
To use this project, follow these steps:

1. Clone the repository to your local machine:
git clone https://github.com/your-username/bully-detection-project.git
## Installation
You can install these dependencies using `pip`. For example:
pip install transformers scikit-learn pandas matplotlib seaborn nltk imbalanced-learn
git clone https://github.com/yourusername/nlp-text-classification.git cd nlp-text-classification
pip install -r requirements.txt

Run the Jupyter Notebook data_preprocessing.ipynb to clean and preprocess the dataset.

The cleaned dataset will be saved as cleaned_2.csv in the project's root directory.

You can then use this cleaned dataset for your analysis or further modeling.

## Project Description
The primary goal of this project is to perform text classification on comments, with a focus on identifying bullying and categorizing comments into three sentiment classes: Negative, Neutral, and Positive. The project involves various steps, including data preprocessing, model training, evaluation, and user input predictions.

## Dependencies
This project relies on the following Python libraries and frameworks:

- TensorFlow
- Transformers (Hugging Face Transformers)
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn
- NLTK
- Imbalanced-learn (for oversampling)

# Data Preprocessing
The project includes data cleaning and preprocessing steps, such as removing stopwords, lemmatization, and tokenization.
The dataset is balanced using oversampling techniques to handle class imbalance.
# Model Training
The BERT (Bidirectional Encoder Representations from Transformers) model is used for text classification.

# Usage
Preprocess the dataset: The dataset should be in a CSV format with columns named "Clean_coment" and "Output". You can customize the preprocessing steps in the provided code.

Model Training: You can train different models (BERT, GRU, LSTM) by running the corresponding scripts. Make sure to adjust hyperparameters and training settings as needed.

Model Evaluation: Evaluate the models using metrics such as accuracy, precision, recall, and F1-score. The evaluation results will be saved, and you can visualize them as needed.

Inference: You can also perform inference on new text data using the trained models.

Models
BERT
BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art NLP model that utilizes pre-trained language representations.
Tokenization and encoding are handled using the Hugging Face Transformers library.
The model is fine-tuned for text classification.
GRU and LSTM
GRU (Gated Recurrent Unit) and LSTM (Long Short-Term Memory) are recurrent neural network (RNN) architectures.
They are used for text classification tasks.
Both models are implemented and compared for performance.
Performance Comparison
We compare the performance of the BERT, GRU, and LSTM models using the following evaluation metrics:

Accuracy
Precision
Recall
F1-score
Performance results are visualized to assess the effectiveness of each model on the text classification task.

Results
We provide the results of the model comparison, including performance metrics and visualizations. These results can be found in the Results directory.
Model training involves fine-tuning a pre-trained BERT model on the dataset.
# Model Evaluation
The project evaluates the model using various metrics, including accuracy, precision, recall, F1-score, and confusion matrices.
Visualizations are provided to understand model performance.
# Predictions and User Input
The model is capable of making predictions on new text data.
User input can be processed and classified into one of the three sentiment classes: Negative, Neutral, or Positive.
