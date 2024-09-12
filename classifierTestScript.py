import re
import torch
import torch.nn as nn
import pickle
from tokenizerFile import tokenizer
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
from tabulate import tabulate

# Load the vocabulary
with open('vocab_py3.pkl', 'rb') as f:  # Ensure this path matches the path used in the training script
    token2idx = pickle.load(f)

# Inverse vocabulary for any other usage if needed
idx2token = {v: k for k, v in token2idx.items()}

# Define the LSTM Model class
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)  # Binary classification

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        if hidden is None:
            out, hidden = self.lstm(x)
        else:
            out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # Use the output of the last time step
        return out, hidden

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(vocab_size=len(token2idx), embedding_dim=32, hidden_dim=32, num_layers=2)
model.load_state_dict(torch.load('lstm_classifier_py3.pth', map_location=device))  # Ensure this path matches the path used in the training script
model.to(device)
model.eval()

def evaluate_model(model, token2idx, input_str, device):
    model.eval()
    tokens = tokenizer(input_str)
    token_indices = [token2idx[token] for token in tokens if token in token2idx]
    if len(token_indices) == 0:
        return 'Invalid'
    seq = torch.tensor(token_indices, dtype=torch.long).unsqueeze(0).to(device) # unsqueeze adds an extra dimension at the beginning to represent the batch size

    with torch.no_grad():
        output, _ = model(seq)
        _, predicted = torch.max(output.data, 1)

    return 'Correct' if predicted.item() == 0 else 'Faulty'

def read_snippets(file_path, label):
    with open(file_path, 'r') as file:
        snippets = file.read().strip().split('-----')
        return [(snippet, label) for snippet in snippets]


def evaluate_accuracy(snippets, model, token2idx, device):
    y_true = []
    y_pred = []

    for snippet, actual_result in snippets:
        predicted_result = evaluate_model(model, token2idx, snippet, device)
        y_true.append(0 if actual_result == 'Correct' else 1)  # 0 for Correct, 1 for Faulty
        y_pred.append(0 if predicted_result == 'Correct' else 1)  # 0 for Correct, 1 for Faulty

    return y_true, y_pred

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
from tabulate import tabulate

def evaluate_accuracy(snippets, model, token2idx, device):
    y_true = []
    y_pred = []

    for snippet, actual_result in snippets:
        predicted_result = evaluate_model(model, token2idx, snippet, device)
        y_true.append(0 if actual_result == 'Correct' else 1)  # 0 for Correct, 1 for Faulty
        y_pred.append(0 if predicted_result == 'Correct' else 1)  # 0 for Correct, 1 for Faulty

    return y_true, y_pred


if __name__ == "__main__":
    try:
        # Read correct and wrong snippets
        correct_snippets = read_snippets('valid_dc_test_new.txt', 'Correct')
        grammar_errors = read_snippets('introduce_grammar_errors.txt', 'Faulty')
        method_decl_errors = read_snippets('missing_method_reference_initialization_only.txt', 'Faulty')
        var_decl_errors = read_snippets('missing_var_reference_initialization_only.txt', 'Faulty')
        method_call_errors = read_snippets('missing_method_reference_usage_only.txt', 'Faulty')
        var_call_errors = read_snippets('missing_var_reference_usage_only.txt', 'Faulty')

        # Evaluate accuracy for each type of snippets
        snippet_types = {
            'Correct Snippets': correct_snippets,
            'Grammar Errors': grammar_errors,
            'Method Declaration Errors': method_decl_errors,
            'Variable Declaration Errors': var_decl_errors,
            'Method Call Errors': method_call_errors,
            'Variable Call Errors': var_call_errors
        }

        total_y_true = []
        total_y_pred = []
        accuracies = {}

        for label, snippets in snippet_types.items():
            y_true, y_pred = evaluate_accuracy(snippets, model, token2idx, device)
            accuracy = accuracy_score(y_true, y_pred)
            accuracies[label] = accuracy

            # Append to total lists for overall confusion matrix
            total_y_true.extend(y_true)
            total_y_pred.extend(y_pred)

        # Calculate overall accuracy
        overall_accuracy = accuracy_score(total_y_true, total_y_pred)
        accuracies['Overall'] = overall_accuracy

        # Create a DataFrame to display the accuracies
        df_accuracies = pd.DataFrame(accuracies, index=['DC Accuracies'])
        print(df_accuracies)

        # df_accuracies['ErrorMethods'] = "dc Accuracies"
        # df_accuracies = df_accuracies[['ErrorMethods'] + [col for col in df_accuracies.columns if col != 'ErrorMethods']]

        # Display the table in a markdown-like format
        table = tabulate(df_accuracies, headers='keys', tablefmt='pipe', showindex=True)
        print(table)

        # Save the DataFrame to a CSV file
        df_accuracies.to_csv('accuracies.csv')

        # Calculate and display overall confusion matrix
        total_cm = confusion_matrix(total_y_true, total_y_pred)

        plt.figure(figsize=(10, 7))
        sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Correct', 'Faulty'], yticklabels=['Correct', 'Faulty'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Overall Confusion Matrix')
        plt.savefig("confusion_matrix_overall.png")
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

