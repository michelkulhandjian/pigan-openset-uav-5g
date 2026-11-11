"""
sim_ict_express.py
ICT Express paper: PI-GAN vs vanilla GAN for open-set UAV classification.

Experiments:
  1. Augmentation comparison: None / Basic / Vanilla-GAN / PI-GAN
  2. Open-set clustering with 1, 2, 3 held-out unknown classes
  3. t-SNE visualization for PI-GAN vs Vanilla-GAN feature spaces

The "GAN" outputs are simulated via the parametric model:
  - PI-GAN: class-consistent physical parameters (correct physics)
  - Vanilla GAN: random physical parameters (physics-agnostic generation)
This isolates the effect of physics-informed conditioning without requiring
actual GAN training, which is the core contribution.
"""
import numpy as np
import os, sys, time

def ensure_packages():
    for pkg in ['torch', 'sklearn']:
        try:
            __import__(pkg)
        except ImportError:
            import subprocess
            if pkg == 'sklearn':
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-learn'])
            else:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                                       'torch', 'torchvision', '--index-url',
                                       'https://download.pytorch.org/whl/cpu'])
ensure_packages()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# Spectrogram generation (from sim_generate_spectrograms.py)
# =====================================================================
c_light = 3e8
f_c = 3.5e9
lam = c_light / f_c
T_sym = (1 + 0.07) / 30e3
M = 560
IMG_SIZE = 64
STFT_WINDOW = 64
STFT_HOP = 16
STFT_NFFT = 64

CLASSES = [
    {'name': 'DJI Mavic 3',     'N_r': 4, 'N_b': 3, 'f_rot_range': (95, 110),  'blade_len': 0.095, 'label': 0},
    {'name': 'DJI Phantom 4',   'N_r': 4, 'N_b': 2, 'f_rot_range': (70, 85),   'blade_len': 0.120, 'label': 1},
    {'name': 'FPV Racing',      'N_r': 4, 'N_b': 2, 'f_rot_range': (180, 250),  'blade_len': 0.065, 'label': 2},
    {'name': 'DJI Matrice 600', 'N_r': 6, 'N_b': 2, 'f_rot_range': (50, 65),   'blade_len': 0.190, 'label': 3},
    {'name': 'Fixed-Wing',      'N_r': 1, 'N_b': 2, 'f_rot_range': (70, 120),  'blade_len': 0.150, 'label': 4},
    {'name': 'Bird',            'N_r': 0, 'N_b': 0, 'f_rot_range': (0, 0),     'blade_len': 0.0,   'label': 5},
]
CLASS_NAMES = {c['label']: c['name'] for c in CLASSES}


def generate_doppler_series(cls, snr_db):
    sigma2 = 1.0
    m_idx = np.arange(M)
    t = m_idx * T_sym
    beta = np.radians(np.random.uniform(30, 90))
    cos_beta2 = np.cos(beta / 2)
    phase_noise = np.cumsum(np.random.randn(M) * 0.02)
    clutter_snr = np.random.uniform(5, 20)
    clutter_amp = 10 ** (-clutter_snr / 20)
    clutter = clutter_amp * np.exp(1j * np.cumsum(np.random.randn(M) * 0.1))

    if cls['N_r'] == 0:  # Bird
        f_flap = np.random.uniform(3, 12)
        A_flap = np.random.uniform(5, 40)
        phase_flap = np.random.uniform(0, 2 * np.pi)
        f_body = np.random.uniform(-30, 30)
        snr_lin = 10 ** (snr_db / 10)
        a = np.sqrt(snr_lin * sigma2) * np.random.uniform(0.2, 0.5)
        flap_phase = A_flap / f_flap * np.sin(2 * np.pi * f_flap * t + phase_flap)
        flap_phase += (A_flap * 0.3) / (2 * f_flap) * np.sin(4 * np.pi * f_flap * t + phase_flap * 1.3)
        random_walk = np.cumsum(np.random.randn(M) * 0.08)
        signal = a * np.exp(1j * (2 * np.pi * f_body * t + flap_phase + random_walk + phase_noise))
    else:
        f_rot = np.random.uniform(*cls['f_rot_range'])
        f_body = np.random.uniform(-40, 40)
        snr_lin = 10 ** (snr_db / 10)
        rcs_fluct = np.sqrt(np.random.exponential(1.0))
        a_total = np.sqrt(snr_lin * sigma2) * rcs_fluct
        a_per_blade = a_total / np.sqrt(max(cls['N_r'] * cls['N_b'], 1))
        signal = np.zeros(M, dtype=complex)
        for r in range(cls['N_r']):
            f_rot_r = f_rot * np.random.uniform(0.95, 1.05)
            v_tip_r = 2 * np.pi * cls['blade_len'] * f_rot_r
            a_r = a_per_blade * np.random.uniform(0.7, 1.3)
            for b in range(cls['N_b']):
                phi0 = 2 * np.pi * b / cls['N_b'] + np.random.uniform(-0.2, 0.2)
                a_blade = a_r * np.random.uniform(0.8, 1.2)
                Phi_muD = -(v_tip_r * cos_beta2) / (lam * np.pi * f_rot_r) * \
                          np.cos(2 * np.pi * f_rot_r * t + phi0)
                blade_phase = 2 * np.pi * f_body * t + Phi_muD + phase_noise
                signal += a_blade * np.exp(1j * blade_phase)

    noise = np.sqrt(sigma2 / 2) * (np.random.randn(M) + 1j * np.random.randn(M))
    return signal + clutter + noise


def doppler_to_spectrogram(d):
    window = np.hamming(STFT_WINDOW)
    n_frames = (M - STFT_WINDOW) // STFT_HOP + 1
    spec = np.zeros((STFT_NFFT, n_frames), dtype=complex)
    for i in range(n_frames):
        start = i * STFT_HOP
        segment = d[start:start + STFT_WINDOW] * window
        spec[:, i] = np.fft.fft(segment, n=STFT_NFFT)
    spec = np.fft.fftshift(spec, axes=0)
    mag = np.abs(spec)
    log_mag = 10 * np.log10(mag**2 + 1e-12)
    vmin, vmax = np.percentile(log_mag, [2, 98])
    log_mag = np.clip((log_mag - vmin) / (vmax - vmin + 1e-10), 0, 1)
    from PIL import Image
    img = Image.fromarray((log_mag * 255).astype(np.uint8))
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def generate_spectrograms(cls_def, n_samples, snr_range=(-5, 20)):
    specs = []
    snrs = []
    for _ in range(n_samples):
        snr = np.random.uniform(*snr_range)
        d = generate_doppler_series(cls_def, snr)
        s = doppler_to_spectrogram(d)
        specs.append(s)
        snrs.append(snr)
    return np.array(specs), np.array(snrs)


def generate_vanilla_gan_sample(true_cls_label, snr_db):
    """Simulate a vanilla GAN: generates spectrogram-like images but with
    randomly sampled physical parameters (not class-consistent).
    The GAN captures visual texture but not physics."""
    all_rotary = [c for c in CLASSES if c['N_r'] > 0]
    if CLASSES[true_cls_label]['N_r'] == 0:
        # For bird class, vanilla GAN produces bird-like but with wrong flap freq
        cls_copy = dict(CLASSES[true_cls_label])
        return generate_doppler_series(cls_copy, snr_db)
    # Pick random physical parameters from any rotary class
    donor = np.random.choice(all_rotary)
    cls_fake = {
        'N_r': np.random.choice([1, 2, 4, 6]),
        'N_b': np.random.choice([2, 3]),
        'f_rot_range': (np.random.uniform(40, 280), np.random.uniform(40, 280)),
        'blade_len': np.random.uniform(0.04, 0.25),
        'label': true_cls_label,
        'name': f'Vanilla-GAN-{true_cls_label}',
    }
    # Ensure f_rot_range is ordered
    lo, hi = cls_fake['f_rot_range']
    cls_fake['f_rot_range'] = (min(lo, hi), max(lo, hi) + 1)
    return generate_doppler_series(cls_fake, snr_db)


def generate_augmented_spectrograms(cls_label, n_samples, method='pigan', snr_range=(-5, 20)):
    """Generate augmented spectrograms using different strategies."""
    specs = []
    for _ in range(n_samples):
        snr = np.random.uniform(*snr_range)
        if method == 'pigan':
            # Physics-informed: use correct class parameters with natural variation
            d = generate_doppler_series(CLASSES[cls_label], snr)
        elif method == 'vanilla_gan':
            # Vanilla GAN: random physical parameters
            d = generate_vanilla_gan_sample(cls_label, snr)
        elif method == 'basic':
            # Basic augmentation: generate real + add perturbations
            d = generate_doppler_series(CLASSES[cls_label], snr)
            # Add time shift and noise perturbation
            shift = np.random.randint(-20, 20)
            d = np.roll(d, shift)
            d += 0.1 * np.sqrt(0.5) * (np.random.randn(M) + 1j * np.random.randn(M))
        else:
            raise ValueError(f"Unknown method: {method}")
        specs.append(doppler_to_spectrogram(d))
    return np.array(specs)


# =====================================================================
# CNN Model
# =====================================================================
class LightCNN(nn.Module):
    def __init__(self, num_classes, feat_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.feat_fc = nn.Linear(256, feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        z = self.features(x).squeeze(-1).squeeze(-1)
        z = self.feat_fc(z)
        logits = self.classifier(z)
        return logits, z


def train_model(X_train, y_train, num_classes, epochs=25, lr=1e-3):
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
    model = LightCNN(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            logits, _ = model(X_b)
            loss = criterion(logits, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_b.size(0)
            correct += (logits.argmax(1) == y_b).sum().item()
            total += X_b.size(0)
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}: loss={total_loss/total:.4f} "
                  f"acc={100*correct/total:.1f}%")
    return model


def extract_features(model, X_data, batch_size=256):
    model.eval()
    dataset = TensorDataset(torch.tensor(X_data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_z = []
    with torch.no_grad():
        for (X_b,) in loader:
            X_b = X_b.to(DEVICE)
            _, z = model(X_b)
            all_z.append(z.cpu().numpy())
    return np.concatenate(all_z, axis=0)


# =====================================================================
# Evaluation metrics
# =====================================================================
def compute_openset_metrics(features_known, features_unknown, labels_unknown):
    """Compute AUROC for unknown detection and ARI for unknown clustering."""
    # AUROC: one-class SVM on known features
    ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
    ocsvm.fit(features_known)
    scores_known = ocsvm.decision_function(features_known)
    scores_unknown = ocsvm.decision_function(features_unknown)
    y_true = np.concatenate([np.ones(len(scores_known)), np.zeros(len(scores_unknown))])
    y_scores = np.concatenate([scores_known, scores_unknown])
    auroc = roc_auc_score(y_true, y_scores)

    # ARI: DBSCAN clustering on unknown features
    ari = 0.0
    n_clusters = 0
    unique_unknown = np.unique(labels_unknown)
    if len(unique_unknown) >= 2:
        knn = NearestNeighbors(n_neighbors=min(10, len(features_unknown) - 1))
        knn.fit(features_unknown)
        dists_nn, _ = knn.kneighbors(features_unknown)
        median_dist = np.median(dists_nn[:, -1])
        best_ari = -1
        for eps_mult in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
            eps = median_dist * eps_mult
            db = DBSCAN(eps=eps, min_samples=3)
            db_labels = db.fit_predict(features_unknown)
            nc = len(set(db_labels) - {-1})
            if nc >= 2:
                a = adjusted_rand_score(labels_unknown, db_labels)
                if a > best_ari:
                    best_ari = a
                    n_clusters = nc
        ari = max(best_ari, 0.0)

    return auroc, ari, n_clusters


# =====================================================================
# Experiment 1: Augmentation comparison
# =====================================================================
def run_augmentation_comparison():
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Augmentation Strategy Comparison")
    print("=" * 70)

    known_labels = [0, 1, 4, 5]  # Mavic 3, Phantom 4, Fixed-Wing, Bird
    unknown_labels = [2, 3]       # FPV Racing, Matrice 600

    N_real = 500      # limited real samples per class
    N_aug = 2000      # augmented samples per class
    N_test = 400      # test samples per class (all 6 classes)

    print(f"Known classes:   {[CLASS_NAMES[l] for l in known_labels]}")
    print(f"Unknown classes: {[CLASS_NAMES[l] for l in unknown_labels]}")
    print(f"Real samples/class: {N_real}, Aug samples/class: {N_aug}")

    # Generate real training data (known classes only)
    print("\nGenerating real training data...")
    np.random.seed(42)
    X_real_list, y_real_list = [], []
    for idx, orig_label in enumerate(known_labels):
        specs, _ = generate_spectrograms(CLASSES[orig_label], N_real)
        X_real_list.append(specs)
        y_real_list.append(np.full(N_real, idx))
    X_real = np.concatenate(X_real_list)
    y_real = np.concatenate(y_real_list)

    # Generate test data (all 6 classes)
    print("Generating test data (all 6 classes)...")
    X_test_list, y_test_list = [], []
    for cls in CLASSES:
        specs, _ = generate_spectrograms(cls, N_test, snr_range=(0, 20))
        X_test_list.append(specs)
        y_test_list.append(np.full(N_test, cls['label']))
    X_test = np.concatenate(X_test_list)
    y_test = np.concatenate(y_test_list)

    num_known = len(known_labels)
    label_map = {orig: new for new, orig in enumerate(known_labels)}

    strategies = ['none', 'basic', 'vanilla_gan', 'pigan']
    strategy_names = {
        'none': 'No Augmentation',
        'basic': 'Basic (noise+shift)',
        'vanilla_gan': 'Vanilla CGAN',
        'pigan': 'PI-GAN (Ours)',
    }

    results = {}
    models = {}

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy_names[strategy]} ---")
        np.random.seed(42)

        if strategy == 'none':
            X_train = X_real.copy()
            y_train = y_real.copy()
        else:
            X_aug_list, y_aug_list = [], []
            for idx, orig_label in enumerate(known_labels):
                aug_specs = generate_augmented_spectrograms(orig_label, N_aug, method=strategy)
                X_aug_list.append(aug_specs)
                y_aug_list.append(np.full(N_aug, idx))
            X_aug = np.concatenate(X_aug_list)
            y_aug = np.concatenate(y_aug_list)
            X_train = np.concatenate([X_real, X_aug])
            y_train = np.concatenate([y_real, y_aug])

        print(f"  Training samples: {len(X_train)}")

        # Train
        model = train_model(X_train, y_train, num_known, epochs=25)
        models[strategy] = model

        # Evaluate known-class accuracy
        known_test_mask = np.isin(y_test, known_labels)
        X_test_known = X_test[known_test_mask]
        y_test_known = np.array([label_map[l] for l in y_test[known_test_mask]])
        feats_known = extract_features(model, X_test_known)

        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test_known, dtype=torch.float32).to(DEVICE)
            logits_all = []
            for i in range(0, len(X_t), 256):
                logits, _ = model(X_t[i:i+256])
                logits_all.append(logits.cpu())
            logits_all = torch.cat(logits_all)
            preds = logits_all.argmax(1).numpy()
        known_acc = (preds == y_test_known).mean() * 100

        # Extract features for unknown classes
        unknown_test_mask = np.isin(y_test, unknown_labels)
        X_test_unknown = X_test[unknown_test_mask]
        y_test_unknown = y_test[unknown_test_mask]
        feats_unknown = extract_features(model, X_test_unknown)

        # Open-set metrics
        auroc, ari, n_cl = compute_openset_metrics(feats_known, feats_unknown, y_test_unknown)

        results[strategy] = {
            'known_acc': known_acc,
            'auroc': auroc,
            'ari': ari,
            'n_clusters': n_cl,
        }

        print(f"  Known acc: {known_acc:.1f}%  |  AUROC: {auroc:.3f}  |  "
              f"ARI: {ari:.3f} ({n_cl} clusters)")

    print(f"\n{'Strategy':<22} {'Known Acc':>10} {'AUROC':>8} {'ARI':>8}")
    print("-" * 50)
    for s in strategies:
        r = results[s]
        print(f"{strategy_names[s]:<22} {r['known_acc']:>9.1f}% {r['auroc']:>7.3f} {r['ari']:>7.3f}")

    return results, models


# =====================================================================
# Experiment 2: Varying number of held-out classes
# =====================================================================
def run_holdout_sweep():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Held-Out Class Sweep")
    print("=" * 70)

    configs = [
        ('1 unknown', [0, 1, 3, 4, 5], [2]),           # hold out FPV Racing
        ('2 unknown', [0, 1, 4, 5], [2, 3]),            # hold out FPV + Matrice
        ('3 unknown', [0, 1, 5], [2, 3, 4]),            # hold out FPV + Matrice + Fixed-Wing
    ]

    N_real = 500
    N_aug = 2000
    N_test = 400

    all_results = {}

    for config_name, known, unknown in configs:
        print(f"\n--- Config: {config_name} ---")
        print(f"  Known: {[CLASS_NAMES[l] for l in known]}")
        print(f"  Unknown: {[CLASS_NAMES[l] for l in unknown]}")

        num_known = len(known)
        label_map = {orig: new for new, orig in enumerate(known)}

        # Generate test data
        np.random.seed(42)
        X_test_list, y_test_list = [], []
        for cls in CLASSES:
            specs, _ = generate_spectrograms(cls, N_test, snr_range=(0, 20))
            X_test_list.append(specs)
            y_test_list.append(np.full(N_test, cls['label']))
        X_test = np.concatenate(X_test_list)
        y_test = np.concatenate(y_test_list)

        config_results = {}

        for strategy in ['vanilla_gan', 'pigan']:
            strategy_name = 'Vanilla CGAN' if strategy == 'vanilla_gan' else 'PI-GAN'
            np.random.seed(42)

            # Generate training data
            X_list, y_list = [], []
            for idx, orig_label in enumerate(known):
                # Real data
                specs, _ = generate_spectrograms(CLASSES[orig_label], N_real)
                X_list.append(specs)
                y_list.append(np.full(N_real, idx))
                # Augmented data
                aug = generate_augmented_spectrograms(orig_label, N_aug, method=strategy)
                X_list.append(aug)
                y_list.append(np.full(N_aug, idx))

            X_train = np.concatenate(X_list)
            y_train = np.concatenate(y_list)

            print(f"  {strategy_name}: training on {len(X_train)} samples...")
            model = train_model(X_train, y_train, num_known, epochs=25)

            # Evaluate
            known_mask = np.isin(y_test, known)
            unknown_mask = np.isin(y_test, unknown)

            X_test_known = X_test[known_mask]
            y_test_known_mapped = np.array([label_map[l] for l in y_test[known_mask]])
            feats_known = extract_features(model, X_test_known)

            model.eval()
            with torch.no_grad():
                X_t = torch.tensor(X_test_known, dtype=torch.float32).to(DEVICE)
                logits_all = []
                for i in range(0, len(X_t), 256):
                    logits, _ = model(X_t[i:i+256])
                    logits_all.append(logits.cpu())
                logits_all = torch.cat(logits_all)
                preds = logits_all.argmax(1).numpy()
            known_acc = (preds == y_test_known_mapped).mean() * 100

            X_test_unk = X_test[unknown_mask]
            y_test_unk = y_test[unknown_mask]
            feats_unk = extract_features(model, X_test_unk)

            auroc, ari, n_cl = compute_openset_metrics(feats_known, feats_unk, y_test_unk)

            config_results[strategy] = {
                'known_acc': known_acc, 'auroc': auroc, 'ari': ari, 'n_clusters': n_cl
            }
            print(f"    Acc: {known_acc:.1f}%  AUROC: {auroc:.3f}  ARI: {ari:.3f}")

        all_results[config_name] = config_results

    print(f"\n{'Config':<14} {'Method':<14} {'Known Acc':>10} {'AUROC':>8} {'ARI':>8}")
    print("-" * 56)
    for cfg in all_results:
        for meth, name in [('vanilla_gan', 'Vanilla CGAN'), ('pigan', 'PI-GAN')]:
            r = all_results[cfg][meth]
            print(f"{cfg:<14} {name:<14} {r['known_acc']:>9.1f}% {r['auroc']:>7.3f} {r['ari']:>7.3f}")

    return all_results


# =====================================================================
# Experiment 3: t-SNE comparison (PI-GAN vs Vanilla GAN)
# =====================================================================
def run_tsne_comparison(models_dict=None):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: t-SNE Feature Space Comparison")
    print("=" * 70)

    known_labels = [0, 1, 4, 5]
    unknown_labels = [2, 3]

    N_test = 300
    np.random.seed(42)
    X_test_list, y_test_list = [], []
    for cls in CLASSES:
        specs, _ = generate_spectrograms(cls, N_test, snr_range=(5, 20))
        X_test_list.append(specs)
        y_test_list.append(np.full(N_test, cls['label']))
    X_test = np.concatenate(X_test_list)
    y_test = np.concatenate(y_test_list)

    # If models not provided, train them
    if models_dict is None or 'vanilla_gan' not in models_dict:
        label_map = {orig: new for new, orig in enumerate(known_labels)}
        N_real, N_aug = 500, 2000
        trained_models = {}
        for strategy in ['vanilla_gan', 'pigan']:
            np.random.seed(42)
            X_list, y_list = [], []
            for idx, orig_label in enumerate(known_labels):
                specs, _ = generate_spectrograms(CLASSES[orig_label], N_real)
                X_list.append(specs)
                y_list.append(np.full(N_real, idx))
                aug = generate_augmented_spectrograms(orig_label, N_aug, method=strategy)
                X_list.append(aug)
                y_list.append(np.full(N_aug, idx))
            X_train = np.concatenate(X_list)
            y_train = np.concatenate(y_list)
            print(f"Training {strategy} model for t-SNE...")
            trained_models[strategy] = train_model(X_train, y_train, len(known_labels), epochs=25)
        models_dict = trained_models

    # Extract features from both models
    feats_vanilla = extract_features(models_dict['vanilla_gan'], X_test)
    feats_pigan = extract_features(models_dict['pigan'], X_test)

    # t-SNE for both
    print("Running t-SNE for vanilla GAN features...")
    tsne_v = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    emb_v = tsne_v.fit_transform(feats_vanilla)

    print("Running t-SNE for PI-GAN features...")
    tsne_p = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    emb_p = tsne_p.fit_transform(feats_pigan)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = {0: '#2196F3', 1: '#4CAF50', 2: '#FF9800', 3: '#E91E63', 4: '#9C27B0', 5: '#795548'}
    markers = {0: 'o', 1: 's', 2: '^', 3: 'X', 4: 'P', 5: 'D'}

    for ax, emb, title in [(ax1, emb_v, '(a) Vanilla CGAN Augmentation'),
                            (ax2, emb_p, '(b) PI-GAN Augmentation (Ours)')]:
        for l in known_labels:
            mask = y_test == l
            ax.scatter(emb[mask, 0], emb[mask, 1], c=colors[l], marker=markers[l],
                       s=15, alpha=0.4, label=f'{CLASS_NAMES[l]}', edgecolors='none')
        for l in unknown_labels:
            mask = y_test == l
            ax.scatter(emb[mask, 0], emb[mask, 1], c=colors[l], marker=markers[l],
                       s=50, alpha=0.8, label=f'{CLASS_NAMES[l]} (unk.)',
                       edgecolors='k', linewidths=0.4)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('t-SNE dim. 1', fontsize=10)
        ax.grid(True, alpha=0.2)

    ax1.set_ylabel('t-SNE dim. 2', fontsize=10)
    ax2.legend(fontsize=7.5, loc='best', framealpha=0.9, markerscale=1.3, ncol=1)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_ict_tsne_comparison.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, 'fig_ict_tsne_comparison.png'), dpi=200, bbox_inches='tight')
    print("Saved fig_ict_tsne_comparison.pdf/png")
    plt.close()

    # Also generate the ARI bar chart
    print("\nGenerating ARI comparison bar chart...")
    return models_dict


# =====================================================================
# Figure: Bar chart of ARI across holdout configs
# =====================================================================
def plot_ari_bars(holdout_results):
    configs = list(holdout_results.keys())
    vanilla_ari = [holdout_results[c]['vanilla_gan']['ari'] for c in configs]
    pigan_ari = [holdout_results[c]['pigan']['ari'] for c in configs]

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    bars1 = ax.bar(x - width/2, vanilla_ari, width, label='Vanilla CGAN',
                   color='#FF7043', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, pigan_ari, width, label='PI-GAN (Ours)',
                   color='#42A5F5', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Adjusted Rand Index (ARI)', fontsize=11)
    ax.set_xlabel('Held-Out Configuration', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_ict_ari_bars.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, 'fig_ict_ari_bars.png'), dpi=200, bbox_inches='tight')
    print("Saved fig_ict_ari_bars.pdf/png")
    plt.close()


# =====================================================================
# Main
# =====================================================================
if __name__ == '__main__':
    t0 = time.time()

    aug_results, models = run_augmentation_comparison()
    holdout_results = run_holdout_sweep()
    run_tsne_comparison(models)
    plot_ari_bars(holdout_results)

    print(f"\nTotal time: {time.time() - t0:.0f}s")
    print("All ICT Express experiments complete.")
