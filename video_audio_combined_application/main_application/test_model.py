import joblib
import numpy as np
import librosa

# Path to your saved model
MODEL_PATH = r"C:\Users\shiva\Downloads\wildlife Project\video_audio_combined_application\main_application\audio_rf_model.joblib"

def extract_features(file_path):
    """Extract MFCC features matching training pipeline."""
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

try:
    # 1. Load model data
    data = joblib.load(MODEL_PATH)
    
    # 2. Extract model safely
    if isinstance(data, dict):
        model = data["model"]
        classes = data.get("classes", getattr(model, "classes_", None))
    else:
        model = data
        classes = getattr(model, "classes_", None)

    print("✅ Model loaded successfully!")
    print(f"📊 Total Classes: {len(classes) if classes is not None else 'Unknown'}")
    print(f"🏷️  Classes list: {list(classes) if classes is not None else 'N/A'}")

    # 3. Test with dummy feature vector
    if hasattr(model, "n_features_in_"):
        dummy_input = np.zeros((1, model.n_features_in_))
        prediction = model.predict(dummy_input)
        print(f"⚡ Dummy Input Prediction Test: SUCCESS (Predicted class: {prediction[0]})")

except Exception as e:
    print(f"❌ Error during model test: {e}")