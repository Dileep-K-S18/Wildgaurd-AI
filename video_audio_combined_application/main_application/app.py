import os
import io
import random
import base64
import pathlib
import threading
import pandas as pd
import numpy as np
from datetime import datetime

# ====================== FIX PATH FOR WINDOWS ======================
plt_posix = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from flask import Flask, render_template, Response, request, redirect, url_for, send_file, flash, jsonify
from ultralytics import YOLO
import cv2
import torch

import sounddevice as sd
import librosa
import joblib
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server thread safety
import matplotlib.pyplot as plt
import seaborn as sns

from gtts import gTTS
import pygame

import pymysql
pymysql.install_as_MySQLdb()
import sqlite3

import time
LAST_EMAIL_TIME = 0
EMAIL_COOLDOWN = 60  # Minimum time (in seconds) between sent emails

import os
from flask import Flask
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')

# Optional: If email sender module exists in project root
try:
    from email_sender import alert_sender
except ImportError:
    alert_sender = None

# ====================== INITIAL SETUP ======================s

# Load Visual Models (YOLO)
MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH)

try:
    yolov5_model = torch.hub.load('.', 'custom', path='last_new.pt', source='local', force_reload=False)
    yolov5_model.conf = 0.10
    yolov5_model.iou = 0.25
except Exception as e:
    print(f"Failed to load YOLOv5 model: {e}")
    yolov5_model = None

# CSV Setup for Visual Detections
CSV_FILE = "detections.csv"
CSV_COLUMNS = ["Date", "Time", "Location", "Type", "SubType", "Confidence"]

if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=CSV_COLUMNS).to_csv(CSV_FILE, index=False)

# Locations
LOCATIONS = [
    "North Ridge Forest", "Misty Creek Grove", "Pine Hollow Zone",
    "Greywood Edge", "Fernvale Thicket", "Elderleaf Basin"
]

# 
# llowed visual classes
VEHICLE_NAMES = {"bicycle", "car", "motorcycle", "bus", "truck", "train", "boat"}
ANIMAL_NAMES = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe","buffalo","rhino"}
HUMAN_NAME = "person"

def is_allowed(name: str) -> bool:
    return name == HUMAN_NAME or name in VEHICLE_NAMES or name in ANIMAL_NAMES

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Camera & threading globals
video_capture = None
monitoring = False
monitor_lock = threading.Lock()
frame_counter = 0

# ====================== LOAD AUDIO MODEL ======================
csv_file2 = "audio_data.csv"
is_listening = False
ENERGY_THRESHOLD = 0.00899 

try:
    saved_data = joblib.load("audio_rf_model.joblib")
    if isinstance(saved_data, dict):
        model2 = saved_data["model"]
        labels2 = saved_data.get("labels", [ 'chainsaw', 'vehicle_engine','AK-12','AK-47','IMI-Desert Eagle','M249','MG-42','MP5','Zastava M92','storm'])
    else:
        model2 = saved_data
        labels2 = ['chainsaw', 'vehicle_engine','AK-12','AK-47','IMI-Desert Eagle','M249','MG-42','MP5','Zastava M92', 'Storm' ]
    print("✅ Loaded Audio Classification Model Successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load audio_rf_model.joblib: {e}")
    model2 = None
    labels2 = []

def extract_features_from_audio(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs, axis=1)

def play_alert_sound(text_prompt):
    """Safely speak alert prompt without file-locking crashes."""
    try:
        tts = gTTS(text=text_prompt, lang='en', slow=False)
        filename = f"output_{random.randint(1000, 9999)}.mp3"
        tts.save(filename)
        
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.quit()

        if os.path.exists(filename):
            os.remove(filename)
    except Exception as e:
        print(f"Audio playback error: {e}")

FIREARM_CLASSES = {'AK-12', 'AK-47', 'IMI Desert Eagle', 'M249', 'MG-42', 'MP5', 'Zastava M92'}

def simplify_label(raw_pred):
    raw_pred_str=str(raw_pred).strip().lower()
    if raw_pred_str == 'engine':
        return 'vehicle_engine'
    return raw_pred_str

# ====================== AUDIO MONITOR THREAD ======================
def audio_monitor2():
    global is_listening

    fs = 22050  
    duration = 3  

    while is_listening:
        try:
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()
            y = recording.flatten().astype(np.float64)
            
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.clip(y, -1.0, 1.0)
            
            try: 
                energy = np.sqrt(np.mean(y ** 2))
            except Exception: 
                energy = 0.00898

            if energy < ENERGY_THRESHOLD:
                continue
            

            if model2 is not None:
                features = extract_features_from_audio(y, fs).reshape(1, -1)
                pred_raw = model2.predict(features)[0]
                
                # Map numeric indices to text labels if necessary
                if isinstance(pred_raw, (int, np.integer)) and len(labels2) > pred_raw:
                    raw_text_label = labels2[pred_raw]
                else:
                    raw_text_label = str(pred_raw)

                    if raw_text_label == "M249":
                        raw_text_label = "vehicle"

                pred = simplify_label(raw_text_label)
                print(f"🎧 Raw: {raw_text_label} -> Cleaned: {pred} | Energy={energy:.6f}")
                

                # Save if prediction is gunshot, chainsaw, or vehicle engine
                if str(pred).lower() not in ["background", "silent", "0"]:
                    now = datetime.now()
                    zone = random.choice(LOCATIONS)
                    new_data = {
                        "Date": now.strftime("%Y-%m-%d"),
                        "Time": now.strftime("%H:%M:%S"),
                        "Sound Detected": pred,
                        "Forest Zone": zone
                    }
                    df = pd.DataFrame([new_data])
                    df.to_csv(csv_file2, mode='a', header=not os.path.exists(csv_file2), index=False)
                    print(f"✅ Logged Audio Event: {pred} at {zone}")

                    bot_response = f"Alert! Detected {pred} sound at {zone}. Please check."

                # ==================== ADD THIS EXACT BLOCK ====================
                global LAST_EMAIL_TIME
                current_time = time.time()

                # Share the 60-second cooldown with video alerts
                if (current_time - LAST_EMAIL_TIME) > EMAIL_COOLDOWN:
                    LAST_EMAIL_TIME = current_time
                    if alert_sender:
                        threading.Thread(
                            target=alert_sender, 
                            args=(bot_response,), 
                            daemon=True
                        ).start()
                # ==============================================================

                    #play_alert_sound(bot_response)

        except Exception as e:
            print("Audio Monitor Loop Error:", e)

# ====================== CAMERA MONITOR FUNCTIONS ======================
def init_camera():
    global video_capture
    if video_capture is None or not video_capture.isOpened():
        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def release_camera():
    global video_capture
    if video_capture is not None:
        video_capture.release()
        video_capture = None

def log_to_csv(row_dict):
    df = pd.DataFrame([row_dict])
    df.to_csv(CSV_FILE, mode='a', header=False, index=False)

def gen_frames():
    global frame_counter
    init_camera()
    while True:
        with monitor_lock:
            running = monitoring

        if not running or video_capture is None or not video_capture.isOpened():
            ret, blank = cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + blank.tobytes() + b'\r\n')
            continue

        success, frame = video_capture.read()
        if not success:
            continue

        frame_counter += 1

        if running:
            results = model.predict(source=frame, conf=0.45, imgsz=640, verbose=False)
            res = results[0] if results else None

            if res and hasattr(res, "boxes") and len(res.boxes) > 0:
                for box in res.boxes:
                    try:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        name = model.names[cls_id].lower()
                    except Exception:
                        continue

                    if not is_allowed(name):
                        continue

                    sub_label = name
                    sub_conf = conf

                    if name in ANIMAL_NAMES and yolov5_model is not None:
                        try:
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0:
                                v5_res = yolov5_model(crop, conf=0.45)
                                df_v5 = v5_res.pandas().xyxy[0]
                                if len(df_v5) > 0:
                                    best = df_v5.iloc[0]
                                    sub_label = str(best["name"])
                                    sub_conf = float(best["confidence"])
                        except Exception as e:
                            print(f"YOLOv5 inference failed: {e}")

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{sub_label} {sub_conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Log safely
                    now = datetime.now()
                    zone1 = random.choice(LOCATIONS)

                    log_to_csv({
                        "Date": now.strftime("%Y-%m-%d"),
                        "Time": now.strftime("%H:%M:%S"),
                        "Location": zone1,
                        "Type": name,
                        "SubType": sub_label,
                        "Confidence": round(sub_conf, 4)
                    })

                    # ==================== ADD THIS EXACT BLOCK ====================
                    bot_response = f"Alert! Detected {sub_label} at {zone1}. Please check."

                    global LAST_EMAIL_TIME
                    current_time = time.time()

                    # Only send an email if 60 seconds have passed since the last one
                    if (current_time - LAST_EMAIL_TIME) > EMAIL_COOLDOWN:
                        LAST_EMAIL_TIME = current_time
                        if alert_sender:
                            threading.Thread(
                                target=alert_sender, 
                                args=(bot_response,), 
                                daemon=True
                            ).start()
                    # ==============================================================

        # Encode and yield
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ====================== FLASK ROUTES ======================

@app.route("/")
def home():
    return render_template("login44.html")

@app.route('/register2', methods=['POST', 'GET'])
def register2():
    return render_template('register44.html')  

@app.route('/login', methods=['POST', 'GET'])
def login():
    return render_template('login44.html') 

@app.route('/logedin', methods=['POST'])
def logedin():
    int_features3 = [str(x) for x in request.form.values()]
    if len(int_features3) < 2:
        return render_template('login44.html', error="Please enter both email and password")
        
    logu, passw = int_features3[0], int_features3[1]
    
    db = sqlite3.connect("ddbb.db")
    cursor = db.cursor()
    cursor.execute("SELECT password FROM user_register WHERE email = ?", (logu,))
    row = cursor.fetchone()
    db.close()

    if row is None:
        return render_template('login44.html', error="User not registered")

    stored_password = row[0]
    if stored_password == passw:
        return render_template('home.html')
    else:
        return render_template('login44.html', error="Incorrect password")

@app.route('/register', methods=['POST'])
def register():
    int_features2 = [str(x) for x in request.form.values()]
    if len(int_features2) < 2:
        return jsonify({'result': 'Invalid inputs'})
        
    logu1, passw1 = int_features2[0], int_features2[1]

    db = sqlite3.connect("ddbb.db")
    cursor = db.cursor()
    try:
        cursor.execute("INSERT INTO user_register (email, password) VALUES (?, ?)", (logu1, passw1))
        db.commit()
    except Exception as e:
        print("Registration error:", e)
        db.rollback()
    finally:
        db.close()

    return render_template('login44.html')

@app.route('/audio_monitor')
def monitor_page2():
    return render_template('audio_monitor.html')

@app.route('/start_monitor2', methods=['POST'])
def start_monitor2():
    global is_listening
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
    fs = 22050
    duration = 1
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    y = recording.flatten()

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(5, 2))
    sns.lineplot(x=np.arange(len(y)), y=y, ax=ax, linewidth=1, color="steelblue")
    ax.set_title("Live Audio Signal", fontsize=10)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return jsonify({"image": f"data:image/png;base64,{graph_url}"})

@app.route('/home2')
def home2():
    return render_template('home1.html')

@app.route("/monitor")
def monitor_page():
    return render_template("monitor.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/start_monitor", methods=["POST"])
def start_monitor():
    global monitoring, frame_counter
    with monitor_lock:
        monitoring = True
    frame_counter = 0
    flash("Monitoring started with YOLOv8 + YOLOv5 wildlife detection.", "success")
    return redirect(url_for("monitor_page"))

@app.route("/stop_monitor", methods=["POST"])
def stop_monitor():
    global monitoring
    with monitor_lock:
        monitoring = False
    release_camera()
    flash("Monitoring stopped.", "info")
    return redirect(url_for("monitor_page"))

@app.route("/download_csv")
def download_csv():
    if not os.path.exists(CSV_FILE):
        flash("No detection data available.", "warning")
        return redirect(url_for("monitor_page"))
    return send_file(CSV_FILE, as_attachment=True, download_name="detections.csv")

@app.route("/upload", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if "image" not in request.files:
            flash("No file part", "error")
            return redirect(request.url)

        file = request.files["image"]
        if file.filename == "":
            flash("No file selected", "error")
            return redirect(request.url)

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        results = model.predict(source=filepath, conf=0.25, imgsz=640, verbose=False)
        res = results[0] if results else None
        detections = []

        if res and hasattr(res, "boxes"):
            for box in res.boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    name = model.names[cls_id].lower()
                except Exception:
                    continue

                if not is_allowed(name):
                    continue

                sub_label = name
                sub_conf = conf

                if name in ANIMAL_NAMES and yolov5_model is not None:
                    try:
                        crop = img[y1:y2, x1:x2]
                        if crop.size > 0:
                            v5_res = yolov5_model(crop)
                            df_v5 = v5_res.pandas().xyxy[0]
                            if len(df_v5) > 0:
                                best = df_v5.iloc[0]
                                sub_label = str(best["name"])
                                sub_conf = float(best["confidence"])
                    except Exception as e:
                        print(f"YOLOv5 failed on upload: {e}")

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{sub_label} {sub_conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                detections.append({'label': sub_label, 'confidence': round(sub_conf, 2)})

        out_path = os.path.join(UPLOAD_FOLDER, "proc_" + file.filename)
        cv2.imwrite(out_path, img)
        return render_template("upload.html", uploaded_image="proc_" + file.filename, detections=detections)

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
