from flask import Flask, render_template, jsonify, send_file
import threading
import sounddevice as sd
import numpy as np
import librosa
import joblib
import pandas as pd
import datetime
import random
import os
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# -------------------- Load Model --------------------
saved_data = joblib.load("audio_rf_model.joblib")
model2 = saved_data["model"]
labels2 = saved_data["labels"]

# -------------------- Globals --------------------
is_listening = False
csv_file2 = "audio_data.csv"
LOCATIONS2 = [
    "North Ridge Forest", "Misty Creek Grove", "Pine Hollow Zone",
    "Greywood Edge", "Fernvale Thicket", "Elderleaf Basin"
]

# 🔸 Threshold for detecting significant sound (adjustable)
ENERGY_THRESHOLD = 0.00899  # increase if still too sensitive

# -------------------- Feature Extraction --------------------
def extract_features_from_audio(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

# -------------------- Monitoring Thread --------------------
def audio_monitor2():
    global is_listening

    fs = 22050  # Sampling rate
    duration = 3  # seconds per chunk

    while is_listening:
        try:
            # Record 3-second chunk
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()
            y = recording.flatten()

            # 🔸 Compute RMS energy
            energy = np.sqrt(np.mean(y ** 2))

            # If sound too low (silent or background noise), skip
            if energy < ENERGY_THRESHOLD:
                print(f"🤫 Silent frame skipped (energy={energy:.6f})")
                continue

            # Extract features and predict
            features = extract_features_from_audio(y, fs).reshape(1, -1)
            pred = model2.predict(features)[0]
            print(f"🎧 Detected sound: {pred} | Energy={energy:.6f}")

            # Save only if not background
            if pred != "background":
                now = datetime.datetime.now()
                zone = random.choice(LOCATIONS2)
                new_data = {
                    "Date": now.strftime("%Y-%m-%d"),
                    "Time": now.strftime("%H:%M:%S"),
                    "Sound Detected": pred,
                    "Forest Zone": zone
                }
                df = pd.DataFrame([new_data])
                df.to_csv(csv_file2, mode='a', header=not os.path.exists(csv_file2), index=False)
                print(f"✅ Logged: {pred} at {zone}")

        except Exception as e:
            print("Error:", e)

# -------------------- Flask Routes --------------------
@app.route('/')
def home():
    return render_template('home1.html')

@app.route('/audio_monitor')
def monitor_page2():
    return render_template('audio_monitor.html')

@app.route('/start_monitor2', methods=['POST'])
def start_monitor2():
    global is_listening
    # Delete old CSV if exists
    if os.path.exists(csv_file2):
        os.remove(csv_file2)

    if not is_listening:
        is_listening = True
        t = threading.Thread(target=audio_monitor2)
        t.daemon = True
        t.start()
    return jsonify({"status": "started"})

@app.route('/stop_monitor2', methods=['POST'])
def stop_monitor2():
    global is_listening
    is_listening = False
    return jsonify({"status": "stopped"})

@app.route('/download_csv2')
def download_csv2():
    if os.path.exists(csv_file2):
        return send_file(csv_file2, as_attachment=True)
    return jsonify({"error": "No data available yet!"})

@app.route('/plot_waveform')
def plot_waveform():
    """Capture short audio and return waveform as base64"""
    fs = 22050
    duration = 1
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    y = recording.flatten()

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(5, 2))
    sns.lineplot(x=np.arange(len(y)), y=y, ax=ax, linewidth=1, color="steelblue")
    ax.set_title("Live Audio Signal", fontsize=10)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()

    # Convert plot to base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return jsonify({"image": f"data:image/png;base64,{graph_url}"})

if __name__ == '__main__':
    app.run(debug=True)
