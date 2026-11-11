# PI-GAN: Physics-Informed Generative Augmentation for Open-Set UAV Classification via 5G Micro-Doppler

This repository contains the simulation code for the paper:

> M. Kulhandjian and L. Aslanyan, "Physics-Informed Generative Augmentation for Open-Set UAV Classification via 5G Micro-Doppler," *ICT Express*, 2026.

## Overview

We propose a physics-informed generative adversarial network (PI-GAN) that conditions spectrogram synthesis on rotor kinematics, forcing the classifier's feature space to encode physically meaningful structure. This enables unsupervised separation of unknown UAV types from passive 5G NR radar micro-Doppler signatures.

The system operates in the 5G NR n78 band (3.5 GHz carrier, 50 MHz bandwidth, 30 kHz subcarrier spacing) and models six target classes: DJI Mavic 3, DJI Phantom 4, FPV Racing drone, DJI Matrice 600, Fixed-Wing UAV, and Bird.

## Key Results

Augmentation strategy comparison with 2 unknown classes held out:

| Strategy | Known Acc. | AUROC | ARI |
|---|---|---|---|
| No Augmentation | 84.6% | 0.452 | 0.970 |
| Basic (noise + shift) | 91.0% | 0.508 | 0.975 |
| Vanilla CGAN | 78.0% | 0.455 | 0.007 |
| **PI-GAN (Ours)** | **91.1%** | **0.557** | **0.965** |

Vanilla CGAN augmentation catastrophically collapses open-set clustering (ARI from 0.970 to 0.007), while PI-GAN preserves it (ARI 0.965). The advantage persists as the number of unknown classes scales from 1 to 3.

## Requirements

- Python 3.8+
- PyTorch 1.12+
- See `requirements.txt` for full dependencies

Install:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate spectrograms

```bash
python simulations/sim_generate_spectrograms.py
```

This generates parametric micro-Doppler spectrograms for all six target classes across SNR values from -5 to 20 dB. Outputs are saved as `.npz` files.

### 2. Run PI-GAN vs. CGAN experiments

```bash
python simulations/sim_ict_express.py
```

This runs the three main experiments from the paper:
- **Experiment 1:** Augmentation comparison (No Aug / Basic / Vanilla CGAN / PI-GAN)
- **Experiment 2:** Scaling to 1, 2, and 3 unknown classes
- **Experiment 3:** t-SNE feature space visualization

Results are printed as LaTeX tables and figures are saved to the `simulations/` directory.

### 3. Train the CNN classifier

```bash
python simulations/sim_train_classifier.py
```

Trains the lightweight 4-layer CNN (422K parameters) with 128-dimensional feature bottleneck.

## Citation

If you use this code, please cite:

```bibtex
@article{kulhandjian2026icte,
  author  = {Kulhandjian, Michel and Aslanyan, Levon},
  title   = {Physics-Informed Generative Augmentation for Open-Set {UAV} Classification via {5G} Micro-{D}oppler},
  journal = {ICT Express},
  year    = {2026},
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
