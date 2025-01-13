'''
AUTHOR: AMPARO GODOY PASTORE
ASSIGNMENT 3: SENTIMENT ANALYSIS
COURSE: NATURAL LANGUAGE PROCESSING - CAP6640
INSTRUCTOR: DINGDING WANG
DATE: NOVEMBER 5TH, 2024 - FALL 2024
'''

from textblob import TextBlob
import os

# Defining functions to perform the task
def load_txt(path):
    """
    Loads and reads all text files in the specified directory.

    Args:
        path (str): The directory path containing text files to be read.

    Returns:
        list of str: A list of strings, each representing the contents of a text file.
    """
    txt = []
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r') as file:
            txt.append(file.read())
    return txt

def analyze_txt(texts, label):
    """
    Analyzes the sentiment of each text in a list and calculates accuracy based on a given label.
    
    Args:
        texts (list of str): A list of strings where each string is a text sample to be analyzed.
        label (str): The expected sentiment label ('pos' or 'neg') for accuracy calculation.

    Returns:
        float: The accuracy of sentiment predictions as a decimal value between 0 and 1, 
               based on the proportion of correctly predicted texts.
    """
    correct = 0
    for t in texts:
        blob = TextBlob(t)
        polarity = blob.sentiment.polarity
        prediction = 'pos' if polarity > 0.1 else 'neg'
        if prediction == label:
            correct += 1
    accuracy = correct / len(texts)
    return accuracy

# Load data
postive_txts = "txt_sentoken/pos"
negative_txts = "txt_sentoken/neg"

pos = load_txt(postive_txts)
neg = load_txt(negative_txts)

pos_samples = len(pos)
neg_samples = len(neg)

# Perform sentiment analysis on the data and get accuracy
pos_accuracy = analyze_txt(pos, 'pos')
neg_accuracy = analyze_txt(neg, 'neg')

print(f"Accuracy on positive texts: {pos_accuracy * 100:.2f}% via {pos_samples} samples")
print(f"Accuracy on negative texts: {neg_accuracy * 100:.2f}% via {neg_samples} samples")
print(f"Overall Accuracy: {(pos_accuracy + neg_accuracy) / 2 * 100:.2f}% via {pos_samples + neg_samples} samples")
