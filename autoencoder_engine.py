"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Neural Channel Autoencoder Engine — Team Shannon                            ║
║  End-to-end learned error correction over noisy channels                     ║
║                                                                              ║
║  Implements:                                                                 ║
║    • Deep autoencoder (encoder → channel → decoder) from scratch             ║
║    • AWGN and Rayleigh fading channel models                                 ║
║    • Curriculum learning with SNR scheduling                                 ║
║    • Classical baselines (Repetition, Hamming(7,4))                          ║
║    • Constellation visualization                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from scipy.special import erfc
import json, time, io, base64, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# ═══════════════════════════════════════════════════════════════════════════════
#  ACTIVATION FUNCTIONS & UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(np.float64)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1.0 - s)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(np.clip(x, -500, 500)) - 1))

def elu_deriv(x, alpha=1.0):
    return np.where(x > 0, 1.0, alpha * np.exp(np.clip(x, -500, 500)))

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def normalize_power(x):
    """Normalize transmitted symbols to unit average power."""
    norm = np.sqrt(np.mean(x ** 2, axis=1, keepdims=True) + 1e-8)
    return x / norm

def one_hot(labels, num_classes):
    oh = np.zeros((len(labels), num_classes))
    oh[np.arange(len(labels)), labels] = 1.0
    return oh


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class DenseLayer:
    """Fully connected layer with He initialization and Adam optimizer."""

    def __init__(self, in_dim, out_dim, activation='elu'):
        scale = np.sqrt(2.0 / in_dim)
        self.W = np.random.randn(in_dim, out_dim) * scale
        self.b = np.zeros((1, out_dim))
        self.activation = activation

        # Adam state
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mb, self.vb = np.zeros_like(self.b), np.zeros_like(self.b)
        self.t = 0

        # Cache for backprop
        self.input = None
        self.pre_act = None

    def forward(self, x):
        self.input = x
        self.pre_act = x @ self.W + self.b
        if self.activation == 'relu':
            return relu(self.pre_act)
        elif self.activation == 'elu':
            return elu(self.pre_act)
        elif self.activation == 'sigmoid':
            return sigmoid(self.pre_act)
        elif self.activation == 'linear':
            return self.pre_act
        elif self.activation == 'softmax':
            return softmax(self.pre_act)
        return self.pre_act

    def backward(self, grad_output, lr=0.001):
        if self.activation == 'relu':
            grad_act = grad_output * relu_deriv(self.pre_act)
        elif self.activation == 'elu':
            grad_act = grad_output * elu_deriv(self.pre_act)
        elif self.activation == 'sigmoid':
            grad_act = grad_output * sigmoid_deriv(self.pre_act)
        elif self.activation == 'softmax' or self.activation == 'linear':
            grad_act = grad_output
        else:
            grad_act = grad_output

        grad_W = self.input.T @ grad_act / len(grad_output)
        grad_b = np.mean(grad_act, axis=0, keepdims=True)
        grad_input = grad_act @ self.W.T

        # Adam update
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        self.mW = beta1 * self.mW + (1 - beta1) * grad_W
        self.vW = beta2 * self.vW + (1 - beta2) * grad_W ** 2
        mW_hat = self.mW / (1 - beta1 ** self.t)
        vW_hat = self.vW / (1 - beta2 ** self.t)
        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)

        self.mb = beta1 * self.mb + (1 - beta1) * grad_b
        self.vb = beta2 * self.vb + (1 - beta2) * grad_b ** 2
        mb_hat = self.mb / (1 - beta1 ** self.t)
        vb_hat = self.vb / (1 - beta2 ** self.t)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

        return grad_input


# ═══════════════════════════════════════════════════════════════════════════════
#  CHANNEL MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def awgn_channel(x, snr_db):
    """Analog White Gaussian Noise channel.
    SNR is Eb/N0 in dB, converted to linear noise variance.
    """
    snr_linear = 10 ** (snr_db / 10.0)
    noise_var = 1.0 / (2.0 * snr_linear)
    noise = np.random.randn(*x.shape) * np.sqrt(noise_var)
    return x + noise

def rayleigh_channel(x, snr_db):
    """Rayleigh flat fading channel with AWGN.
    Models multipath propagation / electromagnetic interference.
    """
    snr_linear = 10 ** (snr_db / 10.0)
    noise_var = 1.0 / (2.0 * snr_linear)
    # Rayleigh fading coefficient (complex, take magnitude)
    h_real = np.random.randn(*x.shape) / np.sqrt(2)
    h_imag = np.random.randn(*x.shape) / np.sqrt(2)
    h = np.sqrt(h_real ** 2 + h_imag ** 2)
    noise = np.random.randn(*x.shape) * np.sqrt(noise_var)
    return h * x + noise


# ═══════════════════════════════════════════════════════════════════════════════
#  NEURAL AUTOENCODER
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralAutoencoder:
    """
    End-to-end neural channel autoencoder.

    Architecture:
        Encoder: k bits → one-hot(2^k) → Dense → ELU → Dense → ELU → Dense(n) → Power Norm
        Channel: AWGN or Rayleigh
        Decoder: n symbols → Dense → ELU → Dense → ELU → Dense(2^k) → Softmax

    Parameters:
        k: message bits
        n: channel uses (codeword length)
        Code rate R = k/n
    """

    def __init__(self, k=4, n=7, channel_type='awgn'):
        self.k = k
        self.n = n
        self.M = 2 ** k           # Number of messages
        self.R = k / n             # Code rate
        self.channel_type = channel_type

        hidden_enc = max(64, self.M * 2)
        hidden_dec = max(64, self.M * 2)

        # Encoder layers
        self.enc1 = DenseLayer(self.M, hidden_enc, 'elu')
        self.enc2 = DenseLayer(hidden_enc, hidden_enc, 'elu')
        self.enc3 = DenseLayer(hidden_enc, n, 'linear')

        # Decoder layers
        self.dec1 = DenseLayer(n, hidden_dec, 'elu')
        self.dec2 = DenseLayer(hidden_dec, hidden_dec, 'elu')
        self.dec3 = DenseLayer(hidden_dec, self.M, 'softmax')

        self.training_log = []

    def encode(self, messages):
        """Encode message indices to transmitted symbols."""
        x = one_hot(messages, self.M)
        x = self.enc1.forward(x)
        x = self.enc2.forward(x)
        x = self.enc3.forward(x)
        x = normalize_power(x)
        return x

    def channel(self, x, snr_db):
        if self.channel_type == 'rayleigh':
            return rayleigh_channel(x, snr_db)
        return awgn_channel(x, snr_db)

    def decode(self, y):
        """Decode received symbols to message probabilities."""
        x = self.dec1.forward(y)
        x = self.dec2.forward(x)
        x = self.dec3.forward(x)
        return x

    def forward(self, messages, snr_db):
        encoded = self.encode(messages)
        received = self.channel(encoded, snr_db)
        decoded = self.decode(received)
        return encoded, received, decoded

    def backward(self, messages, decoded_probs, lr=0.001):
        """Backpropagate cross-entropy loss through decoder and encoder."""
        targets = one_hot(messages, self.M)
        # Cross-entropy gradient at softmax output
        grad = (decoded_probs - targets)

        # Decoder backward
        grad = self.dec3.backward(grad, lr)
        grad = self.dec2.backward(grad, lr)
        grad = self.dec1.backward(grad, lr)

        # Channel is stochastic — use straight-through estimator
        # (gradient passes through channel unchanged)

        # Encoder backward (through power normalization — approx gradient)
        grad = self.enc3.backward(grad, lr)
        grad = self.enc2.backward(grad, lr)
        grad = self.enc1.backward(grad, lr)

    def compute_loss(self, messages, decoded_probs):
        targets = one_hot(messages, self.M)
        return -np.mean(np.sum(targets * np.log(decoded_probs + 1e-10), axis=1))

    def compute_ber(self, messages, decoded_probs):
        predicted = np.argmax(decoded_probs, axis=1)
        # Compute bit errors
        bit_errors = 0
        total_bits = 0
        for i in range(len(messages)):
            xor = int(messages[i]) ^ int(predicted[i])
            bit_errors += bin(xor).count('1')
            total_bits += self.k
        return bit_errors / total_bits if total_bits > 0 else 0

    def compute_bler(self, messages, decoded_probs):
        predicted = np.argmax(decoded_probs, axis=1)
        return np.mean(predicted != messages)

    def train_epoch(self, batch_size, snr_db, lr=0.001):
        messages = np.random.randint(0, self.M, batch_size)
        encoded, received, decoded = self.forward(messages, snr_db)
        loss = self.compute_loss(messages, decoded)
        ber = self.compute_ber(messages, decoded)
        bler = self.compute_bler(messages, decoded)
        self.backward(messages, decoded, lr)
        return loss, ber, bler

    def train_curriculum(self, total_epochs=500, batch_size=256, callback=None):
        """
        Curriculum learning: start with high SNR (easy), gradually decrease.
        This helps the network learn basic structure first, then adapt to noise.

        Schedule:
          Phase 1 (0-20%):   SNR 20→12 dB  (clean signal — learn encoding)
          Phase 2 (20-50%):  SNR 12→5 dB   (moderate noise — refine)
          Phase 3 (50-80%):  SNR 5→1 dB    (hard — build robustness)
          Phase 4 (80-100%): SNR 1→0 dB    (extreme — polish)
        """
        self.training_log = []

        for epoch in range(total_epochs):
            progress = epoch / max(total_epochs - 1, 1)

            # Curriculum SNR schedule
            if progress < 0.2:
                snr = 20.0 - 8.0 * (progress / 0.2)
            elif progress < 0.5:
                snr = 12.0 - 7.0 * ((progress - 0.2) / 0.3)
            elif progress < 0.8:
                snr = 5.0 - 4.0 * ((progress - 0.5) / 0.3)
            else:
                snr = 1.0 - 1.0 * ((progress - 0.8) / 0.2)

            # Learning rate decay
            if progress < 0.3:
                lr = 0.003
            elif progress < 0.6:
                lr = 0.001
            elif progress < 0.85:
                lr = 0.0005
            else:
                lr = 0.0002

            loss, ber, bler = self.train_epoch(batch_size, snr, lr)

            entry = {
                'epoch': epoch,
                'loss': float(loss),
                'ber': float(ber),
                'bler': float(bler),
                'snr_db': float(snr),
                'lr': float(lr),
                'progress': float(progress)
            }
            self.training_log.append(entry)

            if callback:
                callback(entry)

        return self.training_log

    def evaluate_ber_curve(self, snr_range, num_samples=5000):
        """Evaluate BER across SNR range."""
        bers = []
        blers = []
        for snr in snr_range:
            messages = np.random.randint(0, self.M, num_samples)
            _, _, decoded = self.forward(messages, snr)
            bers.append(self.compute_ber(messages, decoded))
            blers.append(self.compute_bler(messages, decoded))
        return np.array(bers), np.array(blers)

    def get_constellation(self, num_points=None):
        """Get the learned constellation points."""
        if num_points is None:
            num_points = self.M
        messages = np.arange(self.M)
        encoded = self.encode(messages)
        return encoded

    def encode_text(self, text):
        """Encode ASCII text through the autoencoder."""
        bits_list = []
        for ch in text:
            byte = ord(ch)
            for i in range(8 // self.k + (1 if 8 % self.k else 0)):
                chunk = (byte >> (i * self.k)) & (self.M - 1)
                bits_list.append(chunk)
        return np.array(bits_list)

    def decode_text(self, symbols):
        """Decode symbols back to ASCII text."""
        chunks_per_char = 8 // self.k + (1 if 8 % self.k else 0)
        chars = []
        for i in range(0, len(symbols), chunks_per_char):
            byte = 0
            for j in range(chunks_per_char):
                if i + j < len(symbols):
                    byte |= (int(symbols[i + j]) & (self.M - 1)) << (j * self.k)
            byte = byte & 0xFF
            if 32 <= byte <= 126:
                chars.append(chr(byte))
            else:
                chars.append('·')
        return ''.join(chars)

    def transmit_text(self, text, snr_db):
        """Full pipeline: text → encode → channel → decode → text."""
        messages = self.encode_text(text)
        encoded = self.encode(messages)
        received = self.channel(encoded, snr_db)
        decoded_probs = self.decode(received)
        recovered = np.argmax(decoded_probs, axis=1)
        return self.decode_text(messages), self.decode_text(recovered), messages, recovered


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASSICAL BASELINES
# ═══════════════════════════════════════════════════════════════════════════════

def uncoded_bpsk_ber(snr_db):
    """Theoretical BER for uncoded BPSK over AWGN."""
    snr = 10 ** (snr_db / 10.0)
    return 0.5 * erfc(np.sqrt(snr))

def uncoded_bpsk_ber_rayleigh(snr_db):
    """Theoretical BER for uncoded BPSK over Rayleigh fading."""
    snr = 10 ** (snr_db / 10.0)
    return 0.5 * (1.0 - np.sqrt(snr / (1.0 + snr)))

def repetition_code_ber(snr_db, n_rep=3, channel='awgn'):
    """BER for repetition code with majority voting."""
    if channel == 'awgn':
        pe = uncoded_bpsk_ber(snr_db)
    else:
        pe = uncoded_bpsk_ber_rayleigh(snr_db)
    pe = np.clip(pe, 0, 0.5)
    # Majority voting for n_rep repetitions
    ber = np.zeros_like(snr_db, dtype=float)
    for i in range(n_rep // 2 + 1, n_rep + 1):
        from scipy.special import comb
        ber += comb(n_rep, i, exact=True) * pe**i * (1 - pe)**(n_rep - i)
    return ber

def hamming74_ber(snr_db, channel='awgn'):
    """Approximate BER for Hamming(7,4) code."""
    if channel == 'awgn':
        pe = uncoded_bpsk_ber(snr_db)
    else:
        pe = uncoded_bpsk_ber_rayleigh(snr_db)
    pe = np.clip(pe, 0, 0.5)
    # Hamming(7,4) can correct 1 error
    # Approximate: P_block ≈ 1 - (1-pe)^7 - 7*pe*(1-pe)^6
    p_block = 1.0 - (1 - pe)**7 - 7 * pe * (1 - pe)**6
    return p_block * (3.0 / 7.0)  # Approximate bit error rate


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION ENGINE (Matplotlib → Base64)
# ═══════════════════════════════════════════════════════════════════════════════

# Color scheme
COLORS = {
    'bg': '#0a0e17',
    'surface': '#111827',
    'surface2': '#1a2332',
    'accent1': '#00d4ff',    # Cyan
    'accent2': '#ff6b6b',    # Coral
    'accent3': '#ffd93d',    # Gold
    'accent4': '#6bcb77',    # Green
    'accent5': '#a78bfa',    # Purple
    'text': '#e5e7eb',
    'text_dim': '#6b7280',
    'grid': '#1f2937',
}

def setup_plot_style():
    plt.rcParams.update({
        'figure.facecolor': COLORS['bg'],
        'axes.facecolor': COLORS['surface'],
        'axes.edgecolor': COLORS['grid'],
        'text.color': COLORS['text'],
        'axes.labelcolor': COLORS['text'],
        'xtick.color': COLORS['text_dim'],
        'ytick.color': COLORS['text_dim'],
        'grid.color': COLORS['grid'],
        'grid.alpha': 0.3,
        'font.family': 'monospace',
    })

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

def plot_training_progress(training_log):
    """Generate training progress visualization."""
    setup_plot_style()
    if not training_log:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.text(0.5, 0.5, 'No training data yet', ha='center', va='center',
                fontsize=16, color=COLORS['text_dim'])
        return fig_to_base64(fig)

    epochs = [e['epoch'] for e in training_log]
    losses = [e['loss'] for e in training_log]
    bers = [max(e['ber'], 1e-6) for e in training_log]
    snrs = [e['snr_db'] for e in training_log]
    lrs = [e['lr'] for e in training_log]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('TRAINING PROGRESS — Curriculum Learning', fontsize=14,
                 color=COLORS['accent1'], fontweight='bold', y=0.98)

    # Loss curve
    ax = axes[0, 0]
    ax.plot(epochs, losses, color=COLORS['accent1'], linewidth=1.5, alpha=0.9)
    ax.set_title('Cross-Entropy Loss', fontsize=11, color=COLORS['accent3'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2)

    # BER curve
    ax = axes[0, 1]
    ax.plot(epochs, bers, color=COLORS['accent2'], linewidth=1.5, alpha=0.9)
    ax.set_title('Bit Error Rate (Training)', fontsize=11, color=COLORS['accent3'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('BER')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2)

    # SNR Schedule
    ax = axes[1, 0]
    ax.fill_between(epochs, snrs, alpha=0.3, color=COLORS['accent5'])
    ax.plot(epochs, snrs, color=COLORS['accent5'], linewidth=1.5)
    ax.set_title('Curriculum SNR Schedule', fontsize=11, color=COLORS['accent3'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('SNR (dB)')
    ax.grid(True, alpha=0.2)
    # Phase annotations
    total = len(epochs)
    phases = [
        (0, 0.2, 'Phase 1\nEasy', COLORS['accent4']),
        (0.2, 0.5, 'Phase 2\nModerate', COLORS['accent3']),
        (0.5, 0.8, 'Phase 3\nHard', COLORS['accent2']),
        (0.8, 1.0, 'Phase 4\nExtreme', COLORS['accent1']),
    ]
    for start, end, label, color in phases:
        ax.axvspan(start * total, end * total, alpha=0.08, color=color)

    # Learning rate
    ax = axes[1, 1]
    ax.plot(epochs, lrs, color=COLORS['accent4'], linewidth=1.5)
    ax.set_title('Learning Rate Schedule', fontsize=11, color=COLORS['accent3'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig_to_base64(fig)

def plot_ber_curves(ae, snr_range, channel_type='awgn'):
    """Generate BER comparison plot."""
    setup_plot_style()

    ber_nn, bler_nn = ae.evaluate_ber_curve(snr_range, num_samples=8000)
    ber_uncoded = uncoded_bpsk_ber(snr_range) if channel_type == 'awgn' \
        else uncoded_bpsk_ber_rayleigh(snr_range)
    ber_rep3 = repetition_code_ber(snr_range, 3, channel_type)
    ber_hamming = hamming74_ber(snr_range, channel_type)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot with markers
    ax.semilogy(snr_range, ber_uncoded, 's-', color=COLORS['text_dim'],
                linewidth=2, markersize=5, label='Uncoded BPSK', alpha=0.7)
    ax.semilogy(snr_range, ber_rep3, 'D--', color=COLORS['accent2'],
                linewidth=2, markersize=5, label='Repetition (3,1)')
    ax.semilogy(snr_range, ber_hamming, '^--', color=COLORS['accent3'],
                linewidth=2, markersize=5, label='Hamming (7,4)')
    ax.semilogy(snr_range, np.maximum(ber_nn, 1e-6), 'o-', color=COLORS['accent1'],
                linewidth=2.5, markersize=6, label=f'Neural ({ae.n},{ae.k})',
                zorder=10)

    ax.set_xlabel('Eb/N0 (dB)', fontsize=12)
    ax.set_ylabel('Bit Error Rate', fontsize=12)
    channel_name = 'AWGN' if channel_type == 'awgn' else 'Rayleigh Fading'
    ax.set_title(f'BER Performance — {channel_name} Channel\n'
                 f'Code Rate R = {ae.k}/{ae.n} = {ae.R:.3f}',
                 fontsize=13, color=COLORS['accent3'], fontweight='bold')
    ax.legend(fontsize=10, facecolor=COLORS['surface2'], edgecolor=COLORS['grid'],
              labelcolor=COLORS['text'])
    ax.grid(True, which='both', alpha=0.2)
    ax.set_ylim([1e-5, 1.0])
    ax.set_xlim([snr_range[0], snr_range[-1]])

    fig.tight_layout()
    return fig_to_base64(fig)


def plot_constellation(ae):
    """Plot the learned signal constellation."""
    setup_plot_style()

    constellation = ae.get_constellation()
    n_dims = constellation.shape[1]

    if n_dims == 1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(constellation[:, 0], np.zeros(ae.M),
                   c=np.arange(ae.M), cmap='plasma', s=120, zorder=5, edgecolors='white', linewidth=0.5)
        for i in range(ae.M):
            bits = format(i, f'0{ae.k}b')
            ax.annotate(bits, (constellation[i, 0], 0.02),
                        ha='center', fontsize=7, color=COLORS['text'])
        ax.set_xlabel('Symbol Value')
        ax.set_title('Learned 1D Constellation', fontsize=13,
                      color=COLORS['accent3'], fontweight='bold')
    elif n_dims >= 2:
        # Project to first 2 dimensions
        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(constellation[:, 0], constellation[:, 1],
                             c=np.arange(ae.M), cmap='plasma', s=150,
                             zorder=5, edgecolors='white', linewidth=0.8)
        for i in range(min(ae.M, 32)):
            bits = format(i, f'0{ae.k}b')
            ax.annotate(bits, (constellation[i, 0] + 0.03, constellation[i, 1] + 0.03),
                        fontsize=7, color=COLORS['text'])

        # Draw unit circle for reference
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), '--', color=COLORS['text_dim'],
                alpha=0.3, linewidth=1)

        ax.set_xlabel('Dimension 1', fontsize=11)
        ax.set_ylabel('Dimension 2', fontsize=11)
        ax.set_title(f'Learned 2D Constellation — ({ae.n},{ae.k}) Autoencoder\n'
                     f'{ae.M} codewords projected to first 2 dimensions',
                     fontsize=12, color=COLORS['accent3'], fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    return fig_to_base64(fig)


def plot_channel_effect(ae, snr_db):
    """Visualize how the channel corrupts the signal."""
    setup_plot_style()

    messages = np.arange(ae.M)
    encoded = ae.encode(messages)
    received = ae.channel(encoded, snr_db)

    if ae.n >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Clean constellation
        ax = axes[0]
        ax.scatter(encoded[:, 0], encoded[:, 1], c=np.arange(ae.M),
                   cmap='plasma', s=100, edgecolors='white', linewidth=0.5)
        ax.set_title('Transmitted (Clean)', fontsize=12, color=COLORS['accent4'])
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal')

        # Noisy constellation (multiple transmissions)
        ax = axes[1]
        for _ in range(50):
            rcv = ae.channel(encoded, snr_db)
            ax.scatter(rcv[:, 0], rcv[:, 1], c=np.arange(ae.M),
                       cmap='plasma', s=10, alpha=0.15)
        ax.scatter(encoded[:, 0], encoded[:, 1], c='white', s=30,
                   marker='x', linewidth=1, zorder=10)
        channel_name = 'AWGN' if ae.channel_type == 'awgn' else 'Rayleigh'
        ax.set_title(f'Received ({channel_name}, SNR={snr_db}dB)', fontsize=12,
                     color=COLORS['accent2'])
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal')
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(encoded[:, 0], np.zeros(ae.M), c=COLORS['accent4'],
                   s=100, label='Transmitted', zorder=5)
        for _ in range(30):
            rcv = ae.channel(encoded, snr_db)
            ax.scatter(rcv[:, 0], np.random.randn(ae.M) * 0.02,
                       c=COLORS['accent2'], s=10, alpha=0.2)
        ax.legend()
        ax.set_title(f'Channel Effect at SNR={snr_db}dB')

    fig.suptitle(f'Channel Visualization — SNR = {snr_db} dB',
                 fontsize=14, color=COLORS['accent1'], fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_architecture_diagram(ae):
    """Create a visual diagram of the autoencoder architecture."""
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis('off')

    blocks = [
        (0.5, 'Message\nBits', f'{ae.k} bits', COLORS['accent4']),
        (2.5, 'One-Hot\nEncoding', f'{ae.M} dim', COLORS['accent4']),
        (4.5, 'Encoder\nDNN', f'{ae.M}→{max(64,ae.M*2)}→{ae.n}', COLORS['accent1']),
        (6.8, 'Power\nNorm', f'‖x‖²=1', COLORS['accent5']),
        (8.8, 'Channel\n' + ae.channel_type.upper(), f'SNR: var', COLORS['accent2']),
        (10.8, 'Decoder\nDNN', f'{ae.n}→{max(64,ae.M*2)}→{ae.M}', COLORS['accent1']),
        (13.0, 'Softmax\nOutput', f'{ae.M} probs', COLORS['accent3']),
        (14.8, 'Decoded\nBits', f'{ae.k} bits', COLORS['accent4']),
    ]

    for x, title, sub, color in blocks:
        rect = FancyBboxPatch((x, 1.5), 1.5, 2.0,
                              boxstyle="round,pad=0.1",
                              facecolor=color + '22',
                              edgecolor=color,
                              linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 0.75, 2.9, title, ha='center', va='center',
                fontsize=9, fontweight='bold', color=color)
        ax.text(x + 0.75, 1.9, sub, ha='center', va='center',
                fontsize=7, color=COLORS['text_dim'])

    # Arrows
    for i in range(len(blocks) - 1):
        x1 = blocks[i][0] + 1.5
        x2 = blocks[i + 1][0]
        ax.annotate('', xy=(x2, 2.5), xytext=(x1, 2.5),
                    arrowprops=dict(arrowstyle='->', color=COLORS['text_dim'],
                                    lw=1.5))

    # Labels
    ax.text(5.5, 0.7, '← ENCODER →', ha='center', fontsize=10,
            color=COLORS['accent1'], fontweight='bold')
    ax.text(11.5, 0.7, '← DECODER →', ha='center', fontsize=10,
            color=COLORS['accent1'], fontweight='bold')
    ax.text(8.8, 0.7, 'NOISY\nCHANNEL', ha='center', fontsize=9,
            color=COLORS['accent2'], fontweight='bold')

    ax.text(8.0, 4.5,
            f'Neural Channel Autoencoder ({ae.n},{ae.k})  —  Rate R = {ae.R:.3f}',
            ha='center', va='center', fontsize=14,
            color=COLORS['accent3'], fontweight='bold')

    fig.tight_layout()
    return fig_to_base64(fig)
