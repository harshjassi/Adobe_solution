from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class SimpleANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train_ann(csv_path, model_path, resume=False):
    df = pd.read_csv(csv_path).dropna()
    assert all(col in df.columns for col in ['text', 'label', 'page_number']), "Missing required columns."

    y = df['label']
    drop_cols = ['text', 'label', 'page_number']
    if 'distance_from_top' in df.columns:
        drop_cols.append('distance_from_top')

    X = df.drop(columns=drop_cols)

    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=16, shuffle=True)

    # Model
    model = SimpleANN(X.shape[1])
    if resume and os.path.exists(model_path):
        print("📦 Resuming training from saved model...")
        model.load_state_dict(torch.load(model_path))

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(30):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor)
            val_loss = criterion(val_preds, y_val_tensor).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

        print(f"Epoch {epoch} - Train Loss: {total_loss:.4f} - Val Loss: {val_loss:.4f}")

    if best_state:
        torch.save(best_state, model_path)
        print(f"✅ Best model saved to {model_path} with Val Loss = {best_val_loss:.4f}")

def predict_headings(csv_path, model_path):
    df = pd.read_csv(csv_path).dropna()

    assert 'text' in df.columns and 'page_number' in df.columns, "Missing text or page_number columns"

    texts = df['text'].tolist()
    pages = df['page_number'].tolist()
    distance_from_top = df['distance_from_top'].tolist()
    drop_cols = ['text', 'page_number','distance_from_top']
    if 'label' in df.columns:
        drop_cols.append('label')
    X = df.drop(columns=drop_cols)
    X_tensor = torch.tensor(X.values, dtype=torch.float32)

    model = SimpleANN(X.shape[1])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        preds = model(X_tensor).numpy().flatten()

    headings = []
    for i, p in enumerate(preds):
        if p > 0.5:
            row = X.iloc[i]
            headings.append((texts[i], row['x0'], row['font_size'], row['bold'],pages[i],distance_from_top[i]))

    return headings
