"""
sim_train_classifier.py
Train multi-task CNN: Lightweight CNN backbone + Classification head + Deep SVDD.
Optimized for CPU — uses 64x64 input directly, no upscaling.
"""
import numpy as np
import os, sys, time

def ensure_packages():
    for pkg in ['torch', 'torchvision']:
        try:
            __import__(pkg)
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                                   'torch', 'torchvision', '--index-url',
                                   'https://download.pytorch.org/whl/cpu'])
ensure_packages()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-5
ALPHA_SVDD = 0.1
NUM_CLASSES = 6
FEAT_DIM = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LightCNN(nn.Module):
    """Lightweight CNN for 64x64 spectrograms. ~200K params, fast on CPU."""
    def __init__(self, num_classes=6, feat_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 1x1
        )
        self.feat_fc = nn.Linear(256, feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.svdd_center = None

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        z = self.features(x).squeeze(-1).squeeze(-1)
        z = self.feat_fc(z)
        logits = self.classifier(z)
        return logits, z

    def init_svdd_center(self, dataloader):
        self.eval()
        all_z = []
        with torch.no_grad():
            for X_batch, _ in dataloader:
                X_batch = X_batch.to(DEVICE)
                _, z = self(X_batch)
                all_z.append(z)
        all_z = torch.cat(all_z, dim=0)
        self.svdd_center = all_z.mean(dim=0).detach()
        print(f"SVDD center initialized: norm={self.svdd_center.norm():.2f}")

def load_data(data_dir):
    splits = {}
    for name in ['train', 'val', 'test']:
        path = os.path.join(data_dir, f'spectrograms_{name}.npz')
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run sim_generate_spectrograms.py first.")
            sys.exit(1)
        data = np.load(path)
        X = torch.tensor(data['X'], dtype=torch.float32)
        y = torch.tensor(data['y'], dtype=torch.long)
        snr = data['snr']
        splits[name] = {'X': X, 'y': y, 'snr': snr}
        print(f"Loaded {name}: {X.shape[0]} samples, shape={X.shape[1:]}")
    return splits

def train():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 60)
    print("Training Lightweight CNN Classifier + Deep SVDD")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LR}")
    print(f"Feature dim: {FEAT_DIM}, SVDD alpha: {ALPHA_SVDD}")
    print()

    splits = load_data(data_dir)
    train_ds = TensorDataset(splits['train']['X'], splits['train']['y'])
    val_ds = TensorDataset(splits['val']['X'], splits['val']['y'])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = LightCNN(NUM_CLASSES, FEAT_DIM).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    init_loader = DataLoader(train_ds, batch_size=512, shuffle=False, num_workers=0)
    model.init_svdd_center(init_loader)

    criterion_cls = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits, z = model(X_batch)
            loss_cls = criterion_cls(logits, y_batch)
            loss_svdd = torch.mean((z - model.svdd_center) ** 2)
            loss = loss_cls + ALPHA_SVDD * loss_svdd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += X_batch.size(0)
        scheduler.step()
        train_loss = total_loss / total
        train_acc = 100 * correct / total

        model.eval()
        val_loss_sum, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                logits, z = model(X_batch)
                loss_cls = criterion_cls(logits, y_batch)
                loss_svdd = torch.mean((z - model.svdd_center) ** 2)
                loss = loss_cls + ALPHA_SVDD * loss_svdd
                val_loss_sum += loss.item() * X_batch.size(0)
                val_correct += (logits.argmax(dim=1) == y_batch).sum().item()
                val_total += X_batch.size(0)
        val_loss = val_loss_sum / val_total
        val_acc = 100 * val_correct / val_total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:3d}/{EPOCHS}: loss={train_loss:.4f} acc={train_acc:.1f}% "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.1f}% [{elapsed:.0f}s]", flush=True)

    model_path = os.path.join(data_dir, 'classifier_resnet18.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'svdd_center': model.svdd_center,
        'history': history,
        'feat_dim': FEAT_DIM,
    }, model_path)
    print(f"\nModel saved to {model_path}")
    print(f"Total training time: {time.time()-t0:.0f}s")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Val')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Training Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Val')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)'); ax2.set_title('Classification Accuracy')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(data_dir, 'training_curves.png'), dpi=200)
    plt.close()
    print("Training curves saved.")

if __name__ == '__main__':
    train()
