# Setup Guide — Neural Channel Autoencoder

## Team Shannon | AI Era Course Project

This was a project for Dr. Donna Schaffer's IT 797 "The AI Era" course at Marymount University (Arlington, Virginia, USA) for the Spring 2026 semester.

Lead Developer:  Michael Raskovski

Developer: Maranda Xiong         

Tester: Callistus Onwuka

Paper Editor: John Zehnpfennig II

Videographer: Joseph Broghamer

# Team Shannon — Neural Channel Autoencoder
## AI Era Course Project: Learned Error Correction Over Noisy Channels

---

## Overview

This project implements a **neural autoencoder** that learns to transmit information
reliably over noisy communication channels. Instead of using hand-designed error
correction codes (Hamming, turbo, LDPC), we train a deep neural network to
**discover its own encoding and decoding strategies** end-to-end.

The system jointly optimizes:
- **Encoder**: Maps messages to channel-optimal codewords
- **Decoder**: Recovers messages from noisy received signals

## Original AI Components

1. **Custom Neural Autoencoder** — Implemented entirely from scratch using NumPy
   (no PyTorch/TensorFlow dependency). Dense layers with ELU activations, Adam
   optimizer, He initialization.

2. **Curriculum Learning** — 4-phase SNR-scheduled training:
   - Phase 1 (0-20%):  High SNR (20→12 dB) — learn encoding structure
   - Phase 2 (20-50%): Medium SNR (12→5 dB) — refine representations
   - Phase 3 (50-80%): Low SNR (5→1 dB)  — build robustness
   - Phase 4 (80-100%): Very low SNR (1→0 dB) — extreme conditions

3. **Dual Channel Models**:
   - AWGN: Additive White Gaussian Noise (thermal noise)
   - Rayleigh Fading: Multipath propagation / electromagnetic interference

4. **Straight-Through Gradient Estimator** — Enables backpropagation through
   the stochastic channel layer.

## Architecture

```
Message (k bits) → One-Hot(2^k) → [Encoder DNN] → Power Norm → CHANNEL → [Decoder DNN] → Softmax → Decoded (k bits)
                                                                   ↑
                                                            AWGN or Rayleigh
```

- Encoder: Dense(M→H, ELU) → Dense(H→H, ELU) → Dense(H→n, Linear) → PowerNorm
- Decoder: Dense(n→H, ELU) → Dense(H→H, ELU) → Dense(H→M, Softmax)

Where M = 2^k (message space), H = max(64, 2M) (hidden size), n = channel uses.

## How to Run

### Requirements
- Python 3.8+
- NumPy, SciPy, Matplotlib, Flask

### Launch
```bash
cd 'directory where file is located'
python app.py
```
Then open http://localhost:5010 in your browser.

### Usage Flow
1. **Architecture Tab**: Configure (k, n) and channel type, view system diagram
2. **Train Tab**: Train the autoencoder with curriculum learning
3. **Results Tab**: Generate BER curves, constellation plots, channel visualizations
4. **Live Demo Tab**: Transmit text through the trained system and see error correction

## Configurable Parameters

| Parameter     | Description                     | Options             |
|---------------|---------------------------------|---------------------|
| k             | Message bits                    | 2, 3, 4, 5          |
| n             | Channel uses (codeword length)  | 2, 4, 7, 8, 14      |
| Channel       | Noise model                     | AWGN, Rayleigh       |
| Epochs        | Training iterations             | 50 — 2000            |
| Batch Size    | Samples per training step       | 128, 256, 512, 1024  |

## Baselines

The system compares against three classical baselines:
- **Uncoded BPSK**: No error correction (theoretical BER)
- **Repetition (3,1)**: Simple repetition code with majority voting
- **Hamming (7,4)**: Single error-correcting code

## Key References

- T. O'Shea & J. Hoydis, "An Introduction to Deep Learning for the Physical Layer," 2017
- S. Gruber et al., "On Deep Learning-Based Channel Decoding," 2017
- C. Shannon, "A Mathematical Theory of Communication," 1948

## Project Structure

```
shannon_project/
├── app.py                  # Flask web server
├── autoencoder_engine.py   # Neural autoencoder + training + visualization
├── templates/
│   └── index.html          # Web UI (HTML/CSS/JS)
└── README.md               # This file
```

---
**Team Shannon** | AI Era Course | 2026
