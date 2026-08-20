import asyncio
import os
import json
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from contextlib import asynccontextmanager
from model_utils import load_model, predict_from_raw_audio

model_bundle = None
model_lock = asyncio.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_bundle
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.pkl")
    
    if os.path.exists(MODEL_PATH):
        model_bundle = load_model(MODEL_PATH)
        acc_info = f" (Test Accuracy: {model_bundle.get('accuracy', 0)*100:.1f}%)" if isinstance(model_bundle, dict) and 'accuracy' in model_bundle else ""
        print(f"🧠 Model loaded: {MODEL_PATH}{acc_info}")
    else:
        print("🚨 Model not found. Run 'python trainer.py' first.")
    
    print(f"✅ Backend server started at http://127.0.0.1:8000")
    print("🎧 Ready to receive audio input...")
    yield
    print("🛑 Backend server stopped.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🎧 WebSocket connected.")
    sample_rate = 44100  # Default browser sample rate fallback
    try:
        while True:
            data = await websocket.receive()
            if 'text' in data:
                msg = data['text']
                try:
                    obj = json.loads(msg)
                    if obj.get('type') == 'init':
                        sample_rate = int(obj.get('sampleRate', 44100))
                        await websocket.send_text(json.dumps({'type': 'init_ack'}))
                        continue
                except Exception:
                    await websocket.send_text(json.dumps({'error': 'bad init'}))
                    continue
                    
            if 'bytes' in data:
                raw = data['bytes']
                audio = np.frombuffer(raw, dtype=np.float32)
                if audio.size == 0 or not np.isfinite(audio).all():
                    continue
                    
                async with model_lock:
                    if model_bundle is None:
                        await websocket.send_text(json.dumps({'type': 'error', 'message': 'Model not loaded'}))
                        continue
                    label, conf = predict_from_raw_audio(model_bundle, audio, sr_in=sample_rate, target_sr=22050)
                    
                print(f"🎶 Predicted Note: {label} (confidence: {conf:.2f})")
                await websocket.send_text(json.dumps({'type': 'pred', 'note': label, 'conf': conf}))

    except WebSocketDisconnect:
        print("❌ WebSocket closed by client.")
        return
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
