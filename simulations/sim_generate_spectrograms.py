"""
sim_generate_spectrograms.py  (v2 - realistic)
Generate synthetic micro-Doppler spectrograms with realistic impairments:
  - Low SNR range (-5 to 20 dB)
  - Clutter residuals after imperfect ECA
  - Phase noise and frequency offsets
  - Overlapping blade-pass frequencies between classes
  - Body Doppler variation and aspect angle randomization
  - Amplitude scintillation (Swerling I fluctuation)
"""
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

c = 3e8
f_c = 3.5e9
lam = c / f_c
T_sym = (1 + 0.07) / 30e3
M = 560
IMG_SIZE = 64
STFT_WINDOW = 64
STFT_HOP = 16
STFT_NFFT = 64
SAMPLES_PER_CLASS = 5000

CLASSES = [
    {'name': 'DJI Mavic 3',     'N_r': 4, 'N_b': 3, 'f_rot_range': (95, 110),  'blade_len': 0.095, 'label': 0},
    {'name': 'DJI Phantom 4',   'N_r': 4, 'N_b': 2, 'f_rot_range': (70, 85),   'blade_len': 0.120, 'label': 1},
    {'name': 'FPV Racing',      'N_r': 4, 'N_b': 2, 'f_rot_range': (180, 250),  'blade_len': 0.065, 'label': 2},
    {'name': 'DJI Matrice 600', 'N_r': 6, 'N_b': 2, 'f_rot_range': (50, 65),   'blade_len': 0.190, 'label': 3},
    {'name': 'Fixed-Wing',      'N_r': 1, 'N_b': 2, 'f_rot_range': (70, 120),  'blade_len': 0.150, 'label': 4},
    {'name': 'Bird',            'N_r': 0, 'N_b': 0, 'f_rot_range': (0, 0),     'blade_len': 0.0,   'label': 5},
]

def generate_doppler_series(cls, snr_db):
    sigma2 = 1.0
    m_idx = np.arange(M)
    t = m_idx * T_sym

    # Random bistatic angle per sample (30-90 deg)
    beta = np.radians(np.random.uniform(30, 90))
    cos_beta2 = np.cos(beta / 2)

    # Phase noise (oscillator drift)
    phase_noise = np.cumsum(np.random.randn(M) * 0.02)

    # Clutter residual after imperfect ECA (colored noise)
    clutter_snr = np.random.uniform(5, 20)  # residual clutter 5-20 dB below signal
    clutter_amp = 10 ** (-clutter_snr / 20)
    clutter = clutter_amp * np.exp(1j * np.cumsum(np.random.randn(M) * 0.1))

    if cls['N_r'] == 0:  # Bird
        f_flap = np.random.uniform(3, 12)
        A_flap = np.random.uniform(5, 40)
        phase_flap = np.random.uniform(0, 2 * np.pi)
        f_body = np.random.uniform(-30, 30)
        snr_lin = 10 ** (snr_db / 10)
        a = np.sqrt(snr_lin * sigma2) * np.random.uniform(0.2, 0.5)
        # Irregular flapping with harmonics
        flap_phase = A_flap / f_flap * np.sin(2 * np.pi * f_flap * t + phase_flap)
        flap_phase += (A_flap * 0.3) / (2 * f_flap) * np.sin(4 * np.pi * f_flap * t + phase_flap * 1.3)
        random_walk = np.cumsum(np.random.randn(M) * 0.08)
        signal = a * np.exp(1j * (2 * np.pi * f_body * t + flap_phase + random_walk + phase_noise))
    else:
        f_rot = np.random.uniform(*cls['f_rot_range'])
        v_tip = 2 * np.pi * cls['blade_len'] * f_rot
        f_body = np.random.uniform(-40, 40)
        snr_lin = 10 ** (snr_db / 10)

        # Swerling I amplitude fluctuation
        rcs_fluct = np.sqrt(np.random.exponential(1.0))
        a_total = np.sqrt(snr_lin * sigma2) * rcs_fluct
        a_per_blade = a_total / np.sqrt(cls['N_r'] * cls['N_b'])

        signal = np.zeros(M, dtype=complex)
        for r in range(cls['N_r']):
            # Per-rotor RPM variation (realistic: rotors don't spin at exactly same speed)
            f_rot_r = f_rot * np.random.uniform(0.95, 1.05)
            v_tip_r = 2 * np.pi * cls['blade_len'] * f_rot_r
            # Per-rotor amplitude variation
            a_r = a_per_blade * np.random.uniform(0.7, 1.3)
            for b in range(cls['N_b']):
                phi0 = 2 * np.pi * b / cls['N_b'] + np.random.uniform(-0.2, 0.2)
                # Blade amplitude variation (wear, geometry)
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
    # Normalize per-spectrogram
    vmin, vmax = np.percentile(log_mag, [2, 98])
    log_mag = np.clip((log_mag - vmin) / (vmax - vmin + 1e-10), 0, 1)
    img = Image.fromarray((log_mag * 255).astype(np.uint8))
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0

def generate_dataset():
    print("=" * 60)
    print("Generating Realistic Micro-Doppler Spectrograms (v2)")
    print("=" * 60)
    print(f"Classes: {len(CLASSES)}, Samples/class: {SAMPLES_PER_CLASS}")
    print(f"SNR range: -5 to 20 dB (realistic)")
    print(f"Impairments: phase noise, clutter residuals, Swerling I, RPM variation")
    print()

    all_specs, all_labels, all_snrs = [], [], []
    for cls in CLASSES:
        print(f"Generating {cls['name']}...", end=' ', flush=True)
        for i in range(SAMPLES_PER_CLASS):
            snr_db = np.random.uniform(-5, 20)
            d = generate_doppler_series(cls, snr_db)
            spec = doppler_to_spectrogram(d)
            all_specs.append(spec)
            all_labels.append(cls['label'])
            all_snrs.append(snr_db)
            if (i + 1) % 2500 == 0:
                print(f"{i+1}", end=' ', flush=True)
        print("done")

    X = np.array(all_specs, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    snrs = np.array(all_snrs, dtype=np.float32)

    idx = np.random.permutation(len(X))
    X, y, snrs = X[idx], y[idx], snrs[idx]

    n = len(X)
    n_train, n_val = int(0.8 * n), int(0.1 * n)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    for name, sl in [('train', slice(0, n_train)),
                     ('val', slice(n_train, n_train + n_val)),
                     ('test', slice(n_train + n_val, None))]:
        path = os.path.join(out_dir, f'spectrograms_{name}.npz')
        np.savez_compressed(path, X=X[sl], y=y[sl], snr=snrs[sl])
        print(f"Saved {name}: {X[sl].shape[0]} samples -> {path}")

    # Example spectrograms at different SNRs
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for idx_cls, cls in enumerate(CLASSES):
        ax = axes[idx_cls // 3, idx_cls % 3]
        mask = (y == cls['label'])
        mid_snr = np.abs(snrs[mask] - 10).argmin()
        sample = X[mask][mid_snr]
        ax.imshow(sample, cmap='inferno', aspect='auto', origin='lower')
        ax.set_title(f"{cls['name']} (SNR≈10 dB)", fontsize=11)
        ax.set_xlabel('Time frame')
        ax.set_ylabel('Doppler bin')
    plt.suptitle('Example Micro-Doppler Spectrograms (Realistic, v2)', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'example_spectrograms.png'), dpi=200)
    plt.close()
    print("Done!")

if __name__ == '__main__':
    generate_dataset()
