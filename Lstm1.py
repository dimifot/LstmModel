import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
#from tokenizerFile import tokenizer
from LstmDataset import CodeDataset
from LstmClassifier import LSTMClassifier
import logging.config
import os
from datetime import datetime
from matplotlib import pyplot as plt

# Step 1: Specify the directory and check if it exists
log_directory = "log_files"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)  # Create the directory if it does not exist

# Step 2: Generate a unique filename for each run
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
filename = f"{log_directory}/log_{current_time}.log"

# Step 3: Configure logging to use the new file
logging.basicConfig(filename=filename, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Example usage of the logger
logger = logging.getLogger('ExampleLogger')
logger.info("This log entry will go into a uniquely named file for this run.")


def train_model(train_dataset, val_dataset, model, epochs, batch_size, lr, device, save_path, vocab_path):
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
            batch_size = seq.size(0)
            #hidden = model.init_hidden(batch_size, device)
            seq, labels = seq.to(device), labels.to(device)
            #hidden = tuple([each.data for each in hidden])  #detach hidden states to prevent backpropagating through the entire training history
            optimizer.zero_grad()
            output, _ = model(seq)
            loss = criterion(output, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  #prevent big gradients
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            # print(predicted)
            # print(labels)
            # print("__________")

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        print(f'Epoch: {epoch + 1}/{epochs}, Training Loss: {epoch_loss}')
        logger.info(f'Epoch: {epoch + 1}/{epochs}, Training Loss: {epoch_loss}')

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        val_losses = list()
        with torch.no_grad():
            for seq, labels in val_loader:
                batch_size = seq.size(0)
                #hidden = model.init_hidden(batch_size, device)
                seq, labels = seq.to(device), labels.to(device)
                #hidden = tuple([each.data for each in hidden])
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

        # Save the model if the validation accuracy is the best we've seen so far.
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            torch.save(model.state_dict(), save_path)
            print(f'Model saved with validation accuracy: {accuracy}%')
            logger.info(f'Model saved with validation accuracy: {accuracy}%')


    logger.info("Finished training model.")

    # Plot and save the losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.savefig('loss_plot.png')  # Save the plot to a file

    # Save the trained model and vocabulary
    torch.save(model.state_dict(), save_path)
    with open(vocab_path, 'wb') as f:
        pickle.dump(train_dataset.dataset.token2idx, f)  # Access the underlying dataset's token2idx
    print(f'Model saved to {save_path} and vocabulary saved to {vocab_path}')
    logger.info(f'Model saved to {save_path} and vocabulary saved to {vocab_path}')


# Example Usage
if __name__ == "__main__":
    with open('valid_dc_new.txt', 'r') as file:
        correct_snippets = file.read().strip().split('-----')  # Split by separator to get individual snippets

    with open('invalid_dc_new.txt', 'r') as file:
        faulty_snippets = file.read().strip().split('-----')  # Split by separator to get individual snippets

    seq_length = 60  # Length of each sequence
    snippets = correct_snippets + faulty_snippets
    print(f"correct dt :   {len(correct_snippets)}")
    print(f"false dt : {len(faulty_snippets)}")
    labels = [0] * len(correct_snippets) + [1] * len(faulty_snippets)  # 0 for correct, 1 for faulty
    print(f" the len of the labels {len(labels)}")
    print(f" the len of the snippes {len(snippets)}")
    dataset = CodeDataset(snippets, labels, seq_length)

    # Split data into training and validation sets
    val_split = 0.20
    val_size = int(len(dataset) * val_split)
    print(f"val size is {val_size}")
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    vocab_size = dataset.vocab_size
    embedding_dim = 16
    hidden_dim = 16
    num_layers = 2
    epochs = 100
    batch_size = 64
    lr = 0.001  # Lower learning rate

    #lr>layers>emb_dim, hidden_dim>batch_size>

    logger.info(f'without hidden')
    # logger.info(f'valid:3302')
    # logger.info(f'invalid:3304')
    logger.info(f'Params:')
    logger.info(f'embedding_dim={embedding_dim}')
    logger.info(f'hidden_dim:{hidden_dim}')
    logger.info(f'num_layers:{num_layers}')
    logger.info(f'epochs:{epochs}')
    logger.info(f'batch_size:{batch_size}')
    logger.info(f'lr:{lr}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
    # Load the saved state_dict
    save_path = 'lstm_classifier_dc2.pth'
    vocab_path = 'vocab_dc2.pkl'
    train_model(train_dataset, val_dataset, model, epochs, batch_size, lr, device, save_path, vocab_path)
