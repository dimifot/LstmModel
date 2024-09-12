import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import logging.config
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tokenizerFile import py_tokenizer
from LstmDataset import CodeDataset_py
from LstmClassifier import LSTMClassifier

# Configure logging
log_directory = "log_files"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
filename = f"{log_directory}/log_{current_time}.log"
logging.basicConfig(filename=filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('ExampleLogger')
logger.info("This log entry will go into a uniquely named file for this run.")


def train_model(train_dataset, val_dataset, model, epochs, batch_size, lr, device):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    train_losses = list()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_accuracy = 0.0
    logger.info("Starting LSTM model training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            output, _ = model(seq)
            loss = criterion(output, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        logger.info(f'Epoch: {epoch + 1}/{epochs}, Training Loss: {epoch_loss}')
        print(f'Epoch: {epoch + 1}/{epochs}, Training Loss: {epoch_loss}')

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        val_losses = list()
        with torch.no_grad():
            for seq, labels in val_loader:
                seq, labels = seq.to(device), labels.to(device)
                output, _ = model(seq)
                loss = criterion(output, labels)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        accuracy = 100 * correct / total
        print(f'Epoch: {epoch + 1}/{epochs}, Validation Loss: {val_loss}, Accuracy: {accuracy}%')
        logger.info(f'Epoch: {epoch + 1}/{epochs}, Validation Loss: {val_loss}, Accuracy: {accuracy}%')
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy

    logger.info("Finished training model.")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.savefig('loss_plot.png')
    plt.show()

    return model


def evaluate_model(model, token2idx, input_str, device):
    model.eval()
    tokens = py_tokenizer(input_str)
    token_indices = [token2idx[token] for token in tokens if token in token2idx]
    if len(token_indices) == 0:
        return 'Invalid'
    seq = torch.tensor(token_indices, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        output, _ = model(seq)
        _, predicted = torch.max(output.data, 1)

    return 'Correct' if predicted.item() == 0 else 'Faulty'


def read_snippets(file_path, label):
    with open(file_path, 'r') as file:
        snippets = file.read().strip().split('-----')
        return [(snippet, label) for snippet in snippets]


# Example Usage
if __name__ == "__main__":
    with open('valid_py.txt', 'r') as file:
        correct_snippets = file.read().strip().split('-----')

    with open('invalid_py_fake.txt', 'r') as file:
        faulty_snippets = file.read().strip().split('-----')

    seq_length = 60
    snippets = correct_snippets + faulty_snippets
    labels = [0] * len(correct_snippets) + [1] * len(faulty_snippets)
    dataset = CodeDataset_py(snippets, labels, seq_length)

    val_split = 0.20
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    vocab_size = dataset.vocab_size
    embedding_dim = 32
    hidden_dim = 32
    num_layers = 2
    epochs = 100
    batch_size = 64
    lr = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)

    trained_model = train_model(train_dataset, val_dataset, model, epochs, batch_size, lr, device)

    # Evaluation
    correct_snippets = read_snippets('valid_py_test.txt', 'Correct')
    wrong_snippets = read_snippets('invalid_py_testing_fake.txt', 'Faulty')
    all_snippets = correct_snippets + wrong_snippets

    y_true = []
    y_pred = []
    for snippet, actual_result in all_snippets:
        predicted_result = evaluate_model(trained_model, dataset.token2idx, snippet, device)
        y_true.append(0 if actual_result == 'Correct' else 1)
        y_pred.append(0 if predicted_result == 'Correct' else 1)

    cm = confusion_matrix(y_true, y_pred)
    tick_labels = [['Correct Snippet', 'Faulty Snippet'], ['Correct Snippet', 'Faulty Snippet']]
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=tick_labels[0], yticklabels=tick_labels[1])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix_dc2.png")
    plt.show()