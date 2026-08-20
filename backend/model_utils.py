import os
import re
import numpy as np
import librosa
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

CANONICAL = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Complete mapping from dataset note tokens to 12 canonical pitch classes
NOTE_MAP = {
    # Naturals (An, Bn, etc.)
    'Cn': 'C', 'Dn': 'D', 'En': 'E', 'Fn': 'F', 'Gn': 'G', 'An': 'A', 'Bn': 'B',
    # Flats (Af, Bf, etc.)
    'Db': 'C#', 'Df': 'C#',
    'Eb': 'D#', 'Ef': 'D#',
    'Gb': 'F#', 'Gf': 'F#',
    'Ab': 'G#', 'Af': 'G#',
    'Bb': 'A#', 'Bf': 'A#',
    'Cb': 'B', 'Fb': 'E',
    # Sharps (C#, Cs, etc.)
    'C#': 'C#', 'Cs': 'C#',
    'D#': 'D#', 'Ds': 'D#',
    'F#': 'F#', 'Fs': 'F#',
    'G#': 'G#', 'Gs': 'G#',
    'A#': 'A#', 'As': 'A#',
    # Single letters (when isolated)
    'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 'A': 'A', 'B': 'B'
}

def parse_label_from_filename(fname):
    """
    Extracts canonical pitch class (e.g. 'C', 'C#', 'D'...) from audio filenames.
    Works with patterns like 'piano_2_Af_a_f_00.wav', 'piano_2_An_j_m_00.wav', etc.
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = re.split(r'[_\-]', base)
    
    # 1. Exact match against known two-character pitch tokens (An, Af, Cs, etc.)
    for p in parts:
        if p in NOTE_MAP and len(p) >= 2:
            return NOTE_MAP[p]
            
    # 2. Check uppercase single letter pitches (avoiding lowercase dynamic tokens like 'a', 'd', 'f')
    for p in parts:
        if p in ['C', 'D', 'E', 'F', 'G', 'A', 'B']:
            return p

    # 3. Regex search for note token like An, Af, C#, etc. anywhere in filename
    match = re.search(r'([A-G][n#sfb])', base, re.IGNORECASE)
    if match:
        token = match.group(1)
        token_formatted = token[0].upper() + token[1].lower() if token[1] in 'nsfb' else token.upper()
        if token_formatted in NOTE_MAP:
            return NOTE_MAP[token_formatted]
            
    return None

def extract_features(y, sr, n_mfcc=20):
    """
    Compute 20 MFCCs + deltas + delta-deltas and extract summary statistics (mean & std).
    Returns an 80-dimensional feature vector.
    """
    if len(y) == 0:
        return np.zeros(n_mfcc * 4, dtype=np.float32)
        
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    stat = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        mfcc_delta.mean(axis=1),
        mfcc_delta2.mean(axis=1)
    ]).astype(np.float32)
    
    return stat

def train_and_save_model(dataset_dir, out_path, sr=22050, test_size=0.15, max_samples=None):
    """
    Scans dataset_dir, extracts features, trains a regularized Random Forest classifier,
    evaluates on test split, and saves the bundle to out_path.
    """
    audio_files = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                audio_files.append(os.path.join(root, file))
                
    if not audio_files:
        raise ValueError(f"No audio files found in {dataset_dir}")
        
    if max_samples and max_samples < len(audio_files):
        np.random.seed(42)
        audio_files = list(np.random.choice(audio_files, size=max_samples, replace=False))
        
    logging.info(f"Found {len(audio_files)} audio files. Extracting features...")
    
    X_list = []
    y_list = []
    
    for fpath in tqdm(audio_files, desc="Extracting features"):
        label = parse_label_from_filename(fpath)
        if not label:
            logging.warning(f"Could not parse label from {fpath}, skipping.")
            continue
            
        try:
            y_audio, _ = librosa.load(fpath, sr=sr, mono=True)
            feat = extract_features(y_audio, sr=sr)
            X_list.append(feat)
            y_list.append(label)
        except Exception as e:
            logging.warning(f"Failed to process {fpath}: {e}")
            continue
            
    if not X_list:
        raise ValueError("Feature extraction produced 0 valid samples.")
        
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    logging.info(f"Dataset summary: {len(X)} samples across {len(le.classes_)} classes: {list(le.classes_)}")
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )
    
    logging.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples.")
    
    # Random Forest Classifier with max_depth to prevent 2GB model bloat
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=25,
        min_samples_split=4,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )
    
    logging.info("Training Random Forest Classifier...")
    clf.fit(X_train, y_train)
    
    # Evaluation
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"==========================================")
    logging.info(f"Test Accuracy: {acc * 100:.2f}%")
    logging.info(f"==========================================")
    logging.info("\n" + classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save model bundle with compression to keep file size compact (~30MB vs 2GB)
    model_bundle = {
        'model': clf,
        'le': le,
        'sr': sr,
        'accuracy': float(acc),
        'classes': list(le.classes_)
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    joblib.dump(model_bundle, out_path, compress=3)
    logging.info(f"Model saved successfully to {out_path} (compressed)")
    
    return X, y

def load_model(path):
    return joblib.load(path)

def predict_from_raw_audio(model_bundle, audio, sr_in=None, target_sr=22050):
    """
    Predicts note and confidence from raw 1D float32 audio buffer.
    """
    if sr_in is None or sr_in <= 0:
        sr_in = 44100  # standard browser default
        
    if sr_in != target_sr:
        audio = librosa.resample(audio, orig_sr=sr_in, target_sr=target_sr)
        
    feat = extract_features(audio, target_sr)
    clf = model_bundle['model']
    le = model_bundle['le']
    
    probs = clf.predict_proba([feat])[0]
    idx = int(np.argmax(probs))
    label = le.inverse_transform([idx])[0]
    conf = float(probs[idx])
    
    return label, conf
