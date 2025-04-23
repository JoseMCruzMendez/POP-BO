import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score

#AI generated boilerplate code

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(NeuralNetwork, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)


def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return val_loss / len(val_loader), accuracy


def k_fold_training(dataset, input_size, hidden_sizes, output_size, k=5, epochs=10, batch_size=32, learning_rate=0.001, stratified=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = StratifiedKFold(n_splits=k) if stratified else KFold(n_splits=k)

    labels = [label for _, label in dataset]
    all_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset, labels)):
        #print(f"Starting fold {fold+1}/{k}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = NeuralNetwork(input_size, hidden_sizes, output_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
            #if epoch % 10 == 0:
                #print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")

        all_accuracies.append(val_accuracy)

    #print(f"Mean Accuracy over {k} folds: {sum(all_accuracies) / k:.4f}")
    return all_accuracies


class PandasDataset(Dataset):
    def __init__(self, dataframe, feature_columns, label_column):
        self.features = dataframe[feature_columns].values.astype('float64')
        self.labels = dataframe[label_column].values.astype('int64')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

# Example usage (assuming dataset is a PyTorch Dataset object):
# dataset = ...  # Your dataset here
# k_fold_training(dataset, input_size=784, hidden_sizes=[128, 64], output_size=10, k=5, epochs=10)
