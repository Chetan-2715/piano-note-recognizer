# Harmonium Note Recognizer — Quick start (detailed)

Prereqs
- Download and install Miniconda first:
  1. Download Miniconda for Windows: https://docs.conda.io/en/latest/miniconda.html
  2. Run the installer (Miniconda3-latest-Windows-x86_64.exe)
  3. During installation:
     - Select "Install for all users" or "Just Me" (your choice)
     - Check "Add Miniconda3 to my PATH environment variable"
  4. After installation, close and reopen PowerShell/Command Prompt

Quick Install (After Miniconda Installation)
```bash
# Create conda environment with Python 3.9
conda create -n harmonium python=3.9 -y
conda activate harmonium

# Install all dependencies via conda-forge
conda install -c conda-forge librosa numpy scikit-learn soundfile joblib uvicorn fastapi -y
pip install python-multipart

# Verify conda is working
conda --version

# Verify Python packages
python -c "import librosa, numpy, soundfile, sklearn, fastapi; print('OK')"

# Train the model
cd backend
python trainer.py --dataset ..\notes_dataset --out model.pkl

# Run the server
python app.py
```

Option A — run directly (app.py runs uvicorn in __main__):
```bash
python app.py
```
Option B — run with uvicorn for more control:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
Server listens on ws://localhost:8000/ws

5) Serve the frontend static files
It's best to serve the frontend over HTTP (not file://) so WebSocket and microphone work consistently.

From the project frontend folder:
```bash
cd ../frontend
# Simple Python server
python -m http.server 8080
# then open in browser:
# http://localhost:8080
```
Alternatively use any static server you prefer.

6) Use the app
- Open http://localhost:8080 in your browser.
- Click the central "🎙️ Record" button; allow microphone access when prompted.
- The frontend sends ~0.3s Float32 audio chunks to ws://localhost:8000/ws; predictions appear below the button as comma-separated notes.
- Click the button again to stop.

7) Retraining / dynamic learning
- The backend automatically saves low-confidence audio chunks to `backend/new_samples/` and launches a background retrain (merging dataset + new samples). To force retrain manually:
```bash
# re-run training (from backend/)
python trainer.py --dataset ..\notes_dataset --out model.pkl
```

Conda Terms-of-Service (ToS) error — quick fix
If you see "Terms of Service have not been accepted" run these commands in PowerShell or Anaconda Prompt to accept the channels' ToS, then create the environment:

PowerShell / Anaconda Prompt:
```powershell
# Accept ToS for the three channels reported by conda
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2

# Now create and activate the env
conda create -n harmonium python=3.9 -y
conda activate harmonium

# Install dependencies
conda install -c conda-forge librosa numpy scikit-learn soundfile joblib uvicorn fastapi -y
pip install python-multipart
```

If you prefer not to use conda (venv fallback)
- Important: use Python 3.9 or 3.10 (librosa/numba do not support Python 3.14).
PowerShell:
```powershell
cd "c:\Users\shaba\OneDrive\Desktop\AI HARMONIUM NOTE RECOGNIZER\backend"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install numpy librosa scikit-learn joblib soundfile fastapi uvicorn python-multipart
```

Verify installation
```powershell
# any environment
python -c "import librosa, numpy, soundfile, sklearn, fastapi; print('OK')"
```

Troubleshooting notes
- If conda commands still fail, open the Anaconda Prompt (Start Menu) and run the ToS accept commands there.
- If librosa or soundfile pip install fails due to native libs, install system libraries:
  - Ubuntu: sudo apt-get install -y libsndfile1 ffmpeg
  - macOS (brew): brew install libsndfile ffmpeg
- Use Python 3.9/3.10 if you run into numba/librosa compatibility issues.

librosa install fails because numba isn't yet compatible with Python 3.14
- Symptom: pip tries to build numba and errors with "Cannot install on Python version 3.14.0; only versions >=3.10,<3.14 are supported."
- Fix: use Python 3.10/3.9 via a new venv or conda env, then install librosa.

Option A — Use installed Python 3.10 (Windows)
1. Check versions:
```powershell
python --version
py -0p            # lists installed Pythons
py -3.10 --version  # check if python3.10 is available
```
2. If 3.10 exists, create venv and install:
```powershell
cd "c:\Users\shaba\OneDrive\Desktop\AI HARMONIUM NOTE RECOGNIZER\backend"
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1   # PowerShell
# or in cmd:
# venv\Scripts\activate

pip install --upgrade pip
pip install numpy librosa scikit-learn joblib soundfile fastapi uvicorn python-multipart
```
3. Verify:
```powershell
python -c "import librosa, numpy, soundfile; print('OK')"
```

If you don't have Python 3.10 installed, download the installer:
- Windows installer (3.10.x): https://www.python.org/downloads/release/python-310x/
Install it and re-run the py -3.10 commands above.

Option B — Use conda (if conda available)
```powershell
conda create -n harmonium python=3.9 -y
conda activate harmonium
conda install -c conda-forge librosa numpy scikit-learn joblib pysoundfile libsndfile ffmpeg uvicorn fastapi -y
pip install python-multipart
```
Note: conda package name for soundfile is pysoundfile; libsndfile is the native library.

If conda or pip still fail to install binary deps, prefer the conda approach (Anaconda Prompt) because it resolves native libraries automatically.

Files referenced
- backend/trainer.py — training entrypoint
- backend/model_utils.py — feature extraction + label parser + model save/load
- backend/app.py — FastAPI WebSocket server
- frontend/index.html, frontend/app.js, frontend/styles.css — UI and Web Audio client
