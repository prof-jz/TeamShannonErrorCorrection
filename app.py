"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Team Shannon — Neural Channel Autoencoder                                   ║
║  Flask Web Application                                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from flask import Flask, render_template, request, jsonify, Response
import json, time, threading
import numpy as np
from autoencoder_engine import (
    NeuralAutoencoder, plot_training_progress, plot_ber_curves,
    plot_constellation, plot_channel_effect, plot_architecture_diagram
)

app = Flask(__name__)

# Global state
models = {}
training_state = {
    'active': False,
    'progress': 0,
    'log': [],
    'current_model_key': None
}

def get_model_key(k, n, channel):
    return f"{k}_{n}_{channel}"

def get_or_create_model(k, n, channel):
    key = get_model_key(k, n, channel)
    if key not in models:
        models[key] = NeuralAutoencoder(k=k, n=n, channel_type=channel)
    return models[key], key


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/train', methods=['POST'])
def train():
    data = request.json
    k = int(data.get('k', 4))
    n = int(data.get('n', 7))
    channel = data.get('channel', 'awgn')
    epochs = int(data.get('epochs', 400))
    batch_size = int(data.get('batch_size', 256))

    ae, key = get_or_create_model(k, n, channel)

    if training_state['active']:
        return jsonify({'error': 'Training already in progress'}), 409

    training_state['active'] = True
    training_state['progress'] = 0
    training_state['log'] = []
    training_state['current_model_key'] = key

    def train_thread():
        try:
            def callback(entry):
                training_state['progress'] = entry['progress']
                training_state['log'].append(entry)

            ae.train_curriculum(total_epochs=epochs, batch_size=batch_size,
                               callback=callback)
        finally:
            training_state['active'] = False

    thread = threading.Thread(target=train_thread, daemon=True)
    thread.start()

    return jsonify({'status': 'started', 'model_key': key})


@app.route('/api/training_status')
def training_status():
    log_start = int(request.args.get('from', 0))
    return jsonify({
        'active': training_state['active'],
        'progress': training_state['progress'],
        'log': training_state['log'][log_start:],
        'total': len(training_state['log'])
    })


@app.route('/api/plots/training')
def plot_training():
    key = request.args.get('key')
    if key and key in models:
        img = plot_training_progress(models[key].training_log)
    else:
        img = plot_training_progress(training_state['log'])
    return jsonify({'image': img})


@app.route('/api/plots/ber')
def plot_ber():
    k = int(request.args.get('k', 4))
    n = int(request.args.get('n', 7))
    channel = request.args.get('channel', 'awgn')
    ae, key = get_or_create_model(k, n, channel)
    snr_range = np.arange(-2, 15, 1.0)
    img = plot_ber_curves(ae, snr_range, channel)
    return jsonify({'image': img})


@app.route('/api/plots/constellation')
def plot_const():
    k = int(request.args.get('k', 4))
    n = int(request.args.get('n', 7))
    channel = request.args.get('channel', 'awgn')
    ae, _ = get_or_create_model(k, n, channel)
    img = plot_constellation(ae)
    return jsonify({'image': img})


@app.route('/api/plots/channel')
def plot_chan():
    k = int(request.args.get('k', 4))
    n = int(request.args.get('n', 7))
    channel = request.args.get('channel', 'awgn')
    snr = float(request.args.get('snr', 5))
    ae, _ = get_or_create_model(k, n, channel)
    img = plot_channel_effect(ae, snr)
    return jsonify({'image': img})


@app.route('/api/plots/architecture')
def plot_arch():
    k = int(request.args.get('k', 4))
    n = int(request.args.get('n', 7))
    channel = request.args.get('channel', 'awgn')
    ae, _ = get_or_create_model(k, n, channel)
    img = plot_architecture_diagram(ae)
    return jsonify({'image': img})


@app.route('/api/transmit', methods=['POST'])
def transmit():
    data = request.json
    k = int(data.get('k', 4))
    n = int(data.get('n', 7))
    channel = data.get('channel', 'awgn')
    text = data.get('text', 'Hello, Shannon!')
    snr_db = float(data.get('snr', 7))

    ae, _ = get_or_create_model(k, n, channel)
    original, recovered, msg_syms, rec_syms = ae.transmit_text(text, snr_db)

    # Compute per-character accuracy
    char_results = []
    for i in range(min(len(original), len(recovered))):
        char_results.append({
            'original': original[i],
            'recovered': recovered[i],
            'match': original[i] == recovered[i]
        })

    accuracy = sum(1 for c in char_results if c['match']) / max(len(char_results), 1)

    return jsonify({
        'original': original,
        'recovered': recovered,
        'accuracy': accuracy,
        'char_results': char_results,
        'num_symbols': len(msg_syms),
        'snr_db': snr_db
    })


@app.route('/api/quick_eval', methods=['POST'])
def quick_eval():
    """Quick BER evaluation at a single SNR point."""
    data = request.json
    k = int(data.get('k', 4))
    n = int(data.get('n', 7))
    channel = data.get('channel', 'awgn')
    snr = float(data.get('snr', 7))

    ae, _ = get_or_create_model(k, n, channel)
    messages = np.random.randint(0, ae.M, 2000)
    _, _, decoded = ae.forward(messages, snr)
    ber = ae.compute_ber(messages, decoded)
    bler = ae.compute_bler(messages, decoded)

    return jsonify({
        'ber': float(ber),
        'bler': float(bler),
        'snr_db': snr,
        'num_tested': 2000
    })


if __name__ == '__main__':
    app.run(debug=False, port=5010, host='0.0.0.0')
