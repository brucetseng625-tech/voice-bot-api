
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import tempfile
import os
import time
import soundfile as sf
import torch
from voxcpm import VoxCPM

app = FastAPI(title="VoxCPM Voice Bot API", description="Local AI Voice API for One-Person Company")

# Global model instance (Lazy loading)
model = None
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def get_model():
    global model
    if model is None:
        print(f"[System] Loading VoxCPM model on {DEVICE}...")
        # load_denoiser=False saves VRAM
        model = VoxCPM.from_pretrained("openbmb/VoxCPM2", device=DEVICE, load_denoiser=False)
        print("[System] Model loaded successfully.")
    return model

class TTSRequest(BaseModel):
    text: str
    style: str = "sweet"  # Default to our favorite preset

# Presets Library
PRESETS = {
    "sweet": "(A charming young woman, seductive and coquettish, sweet and soft voice, slow seductive pace, whispering, acting spoiled and clingy)",
    "professional": "(A professional female assistant, clear, crisp, articulate, moderate pace, confident and polite)",
    "casual": "(A friendly young woman, natural, bright, energetic, casual speaking pace)"
}

@app.post("/v1/audio/speech")
async def generate_speech(request: TTSRequest):
    try:
        start_time = time.time()
        current_model = get_model()
        
        # Apply preset
        prefix = PRESETS.get(request.style, "")
        full_text = f"{prefix}{request.text}"
        
        print(f"[API] Generating audio for: '{request.text}' (Style: {request.style})")
        
        # Generate Audio
        # Using standard settings from our successful tests
        wav = current_model.generate(
            text=full_text,
            cfg_value=2.0,
            inference_timesteps=20
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, wav, current_model.tts_model.sample_rate)
            return FileResponse(
                tmp.name, 
                media_type="audio/wav", 
                filename="output.wav",
                headers={"X-Gen-Time": str(time.time() - start_time)}
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "online", 
        "device": DEVICE, 
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
