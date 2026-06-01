from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import tempfile
import os
import time
import soundfile as sf
import torch
import numpy as np
import io
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
    stream: bool = True   # Default to streaming for better UX

# Presets Library
PRESETS = {
    "sweet": "(A charming young woman, seductive and coquettish, sweet and soft voice, slow seductive pace, whispering, acting spoiled and clingy)",
    "professional": "(A professional female assistant, clear, crisp, articulate, moderate pace, confident and polite)",
    "casual": "(A friendly young woman, natural, bright, energetic, casual speaking pace)"
}

async def audio_stream_generator(text, style):
    """Async generator that yields audio chunks from VoxCPM"""
    try:
        current_model = get_model()
        prefix = PRESETS.get(style, "")
        full_text = f"{prefix}{text}"
        
        print(f"[Stream] Generating audio for: '{text}' (Style: {style})")
        
        # VoxCPM generate_streaming yields numpy arrays
        # Note: We need to handle the float32 -> bytes conversion
        for chunk in current_model.generate_streaming(
            text=full_text,
            cfg_value=2.0,
            inference_timesteps=20
        ):
            # chunk is a numpy array of float32
            # Convert to bytes for streaming
            # 48000Hz, Mono, Float32
            if isinstance(chunk, np.ndarray):
                yield chunk.tobytes()
            elif isinstance(chunk, torch.Tensor):
                yield chunk.cpu().numpy().tobytes()
            
    except Exception as e:
        print(f"[Error] Stream generation failed: {e}")
        # We can't yield an error in the middle of a stream easily,
        # but the connection will close.
        return

@app.post("/v1/audio/speech")
async def generate_speech(request: TTSRequest):
    if request.stream:
        return StreamingResponse(
            audio_stream_generator(request.text, request.style),
            media_type="audio/pcm",
            headers={
                "X-Sample-Rate": "48000",
                "X-Sample-Format": "float32",
                "X-Channels": "1"
            }
        )
    else:
        # Fallback to standard file generation (for testing/download)
        try:
            start_time = time.time()
            current_model = get_model()
            prefix = PRESETS.get(request.style, "")
            full_text = f"{prefix}{request.text}"
            
            print(f"[API] Generating file for: '{request.text}'")
            wav = current_model.generate(
                text=full_text,
                cfg_value=2.0,
                inference_timesteps=20
            )
            
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

@app.get("/")
async def index():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    # Allow CORS for local testing if accessed via other ports
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)
