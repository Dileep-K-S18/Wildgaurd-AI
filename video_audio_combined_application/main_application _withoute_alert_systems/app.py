from flask import Flask, render_template, Response, request, redirect, url_for, send_file, flash
from ultralytics import YOLO
import cv2
import os
import random
import pandas as pd
from datetime import datetime
import threading
import torch
import pathlib





# ====================== FIX PATH FOR WINDOWS (ONCE) ======================
plt = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


from flask import Flask, render_template, jsonify, send_file
import threading
import sounddevice as sd
import numpy as np
import librosa
import joblib
import pandas as pd
#import datetime
import random
import os
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns



#from datetime import datetime

import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb
# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

gmail_list=[]
password_list=[]
gmail_list1=[]
password_list1=[]

# ====================== INITIAL SETUP ======================
app = Flask(__name__)
app.secret_key = "replace-with-a-secret-key-123"

# Models (load once)
MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH)

# YOLOv5 custom model (wild animals) - load once
try:
    yolov5_model = torch.hub.load('.', 'custom', path='last_new.pt', source='local', force_reload=False)
    yolov5_model.conf = 0.10
    yolov5_model.iou = 0.25
except Exception as e:
    print(f"Failed to load YOLOv5 model: {e}")
    yolov5_model = None

# CSV Setup
CSV_FILE = "detections.csv"
CSV_COLUMNS = ["Date", "Time", "Location", "Type", "SubType", "Confidence"]

if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=CSV_COLUMNS).to_csv(CSV_FILE, index=False)

# Locations
LOCATIONS = [
    "North Ridge Forest", "Misty Creek Grove", "Pine Hollow Zone",
    "Greywood Edge", "Fernvale Thicket", "Elderleaf Basin"
]

# Allowed classes
VEHICLE_NAMES = {"bicycle", "car", "motorcycle", "bus", "truck", "train", "boat"}
ANIMAL_NAMES = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
HUMAN_NAME = "person"

def is_allowed(name: str) -> bool:
    return name == HUMAN_NAME or name in VEHICLE_NAMES or name in ANIMAL_NAMES

# Upload folder
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Camera & threading
video_capture = None
monitoring = False
monitor_lock = threading.Lock()
frame_counter = 0




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

            #import datetime

            # Save only if not background
            if pred != "background":
                #now = datetime.datetime.now()
                now = datetime.now()
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
    """Safely append one row to CSV without reading full file."""
    df = pd.DataFrame([row_dict])
    df.to_csv(CSV_FILE, mode='a', header=False, index=False)

def gen_frames():
    global frame_counter
    init_camera()
    while True:
        with monitor_lock:
            running = monitoring

        if not running or video_capture is None or not video_capture.isOpened():
            # Yield a blank frame or break
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
            results = model.predict(source=frame, conf=0.25, imgsz=640, verbose=False)
            res = results[0] if results else None

            if res and hasattr(res, "boxes") and len(res.boxes) > 0:
                for box in res.boxes:
                    try:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        name = model.names[cls_id].lower()
                    except:
                        continue

                    if not is_allowed(name):
                        continue

                    sub_label = name
                    sub_conf = conf

                    # If animal → use YOLOv5 for subtype
                    if name in ANIMAL_NAMES and yolov5_model is not None:
                        try:
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0:
                                v5_res = yolov5_model(crop)
                                df_v5 = v5_res.pandas().xyxy[0]
                                if len(df_v5) > 0:
                                    best = df_v5.iloc[0]
                                    sub_label = str(best["name"])
                                    sub_conf = float(best["confidence"])
                                    print(f"Wild animal detected: {sub_label} ({sub_conf:.2f})")
                        except Exception as e:
                            print(f"YOLOv5 inference failed: {e}")

                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{sub_label} {sub_conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Log safely
                    now = datetime.now()

                    log_to_csv({
                        "Date": now.strftime("%Y-%m-%d"),
                        "Time": now.strftime("%H:%M:%S"),
                        "Location": random.choice(LOCATIONS),
                        "Type": name,
                        "SubType": sub_label,
                        "Confidence": round(sub_conf, 4)
                    })

        # Encode and yield
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ====================== ROUTES ======================










@app.route("/")
def home():
    return render_template("login44.html")



@app.route('/register2',methods=['POST','GET'])
def register2():
    return render_template('register44.html')  



@app.route('/login',methods=['POST','GET'])
def login():
    return render_template('login44.html') 


import pickle
@app.route('/logedin',methods=['POST'])
def logedin():
    
    int_features3 = [str(x) for x in request.form.values()]
    print(int_features3)
    logu=int_features3[0]
    passw=int_features3[1]
    

    name =int_features3[0]

    # Save to a file
    with open("name.pkl", "wb") as f:
        pickle.dump(name, f)

   # if int_features2[0]==12345 and int_features2[1]==12345:

    import MySQLdb


# Open database connection
    db = MySQLdb.connect("localhost","root","","ddbb" )

# prepare a cursor object using cursor() method
    cursor = db.cursor()
    cursor.execute("SELECT user FROM user_register")
    result1=cursor.fetchall()
              #print(result1)
              #print(gmail1)
    for row1 in result1:
                      print(row1)
                      print(row1[0])
                      gmail_list.append(str(row1[0]))
                      
                      #gmail_list.append(row1[0])
                      #value1=row1
                      
    print(gmail_list)
    

    cursor1= db.cursor()
    cursor1.execute("SELECT password FROM user_register")
    result2=cursor1.fetchall()
              #print(result1)
              #print(gmail1)
    for row2 in result2:
                      print(row2)
                      print(row2[0])
                      password_list.append(str(row2[0]))
                      
                      #gmail_list.append(row1[0])
                      #value1=row1
                      
    print(password_list)
    print(gmail_list.index(logu))
    print(password_list.index(passw))
    
    if gmail_list.index(logu)==password_list.index(passw):
        return render_template('home.html')
    else:
        return jsonify({'result':'use proper  gmail and password'})
                  
                                               
@app.route('/register',methods=['POST'])
def register():
    

    int_features2 = [str(x) for x in request.form.values()]
    #print(int_features2)
    #print(int_features2[0])
    #print(int_features2[1])
    r1=int_features2[0]
    print(r1)
    
    r2=int_features2[1]
    print(r2)
    logu1=int_features2[0]
    passw1=int_features2[1]
        
    

    

   # if int_features2[0]==12345 and int_features2[1]==12345:

    import MySQLdb


# Open database connection
    db = MySQLdb.connect("localhost","root",'',"ddbb" )

# prepare a cursor object using cursor() method
    cursor = db.cursor()
    cursor.execute("SELECT user FROM user_register")
    result1=cursor.fetchall()
              #print(result1)
              #print(gmail1)
    for row1 in result1:
                      print(row1)
                      print(row1[0])
                      gmail_list1.append(str(row1[0]))
                      
                      #gmail_list.append(row1[0])
                      #value1=row1
                      
    print(gmail_list1)
    if logu1 in gmail_list1:
                      return jsonify({'result':'this gmail is already in use '})  
    else:

                  #return jsonify({'result':'this  gmail is not registered'})
              

# Prepare SQL query to INSERT a record into the database.
                  sql = "INSERT INTO user_register(user,password) VALUES (%s,%s)"
                  val = (r1, r2)
   
                  try:
   # Execute the SQL command
                                       cursor.execute(sql,val)
   # Commit your changes in the database
                                       db.commit()
                  except:
   # Rollback in case there is any error
                                       db.rollback()

# disconnect from server
                  db.close()
                 # return jsonify({'result':'succesfully registered'})
                  return render_template('login44.html')


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
                except:
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

                label = f"{sub_label} {sub_conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                detections.append({"label": label})

        output_path = os.path.join("static", "output.png")
        cv2.imwrite(output_path, img)

        return render_template("result.html",
                               img_path=output_path,
                               detections=detections,
                               detected_any=len(detections) > 0)

    return render_template("upload.html")

@app.route("/shutdown_camera")
def shutdown_camera():
    release_camera()
    return "Camera released"

# ====================== RUN ======================
if __name__ == "__main__":
    # Optional: suppress FutureWarning
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    import numpy as np  # for blank frame
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)