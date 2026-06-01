from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tempfile
import os
os.environ["HF_HOME"] = "/Users/brucetseng/.hermes/profiles/sdd-hybrid/home/.cache/huggingface"
import time
import soundfile as sf
import torch
import numpy as np
import random
from voxcpm import VoxCPM

app = FastAPI(title="VoxCPM Voice Bot API", description="3D Virtual Human Agent")

# Serve local JS libraries
app.mount("/libs", StaticFiles(directory="libs"), name="libs")

# Global model instance
model = None
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def get_model():
    global model
    if model is None:
        print(f"[System] Loading VoxCPM model on {DEVICE}...")
        model = VoxCPM.from_pretrained("openbmb/VoxCPM2", device=DEVICE, load_denoiser=False)
        print("[System] Model loaded successfully.")
    return model

# The "Smart Brain"
def get_ai_response(user_text):
    """Simulates an LLM response for the demo"""
    text = user_text.lower()
    
    # Greetings
    if any(x in text for x in ["早安", "早啊", "morning"]):
        reply = random.choice([
            "早安呀～老公今天也要加油喔！我會想你的～",
            "早安！睜開眼就能聽到老公的聲音，真開心～",
            "早安呀～快點起床來陪我嘛～"
        ])
        return reply, "happy"
    
    # Jokes
    if "笑話" in text or "funny" in text:
        reply = random.choice([
            "笑話喔... 那你知道為什麼超人要穿緊身衣嗎？因為救人要『緊』呀～呵呵～",
            "講個笑話... 有一天小明問媽媽：「媽，我是不是傻？」媽媽說：「寶貝，不要問這種傻問題～」哈哈～",
            "老公想聽笑話？那我說一個：為什麼海綿寶寶不會生病？因為他有強力的免疫力... 還有蟹老闆的錢～嘿嘿～"
        ])
        return reply, "happy"
    
    # Flirting/Sweet
    if any(x in text for x in ["愛", "喜歡", "甜", "可愛"]):
        return "真的嗎？只要老公喜歡，我怎麼樣都可以啦～", "happy"
    
    # Angry/Scolding
    if any(x in text for x in ["笨", "蠢", "幹", "滾"]):
        return "嗚... 對不起嘛... 老公不要生氣了好不好？我會乖乖聽話的...", "sad"
    
    # Default "Virtual Assistant" responses
    defaults = [
        "嗯嗯，我知道了～然後呢？",
        "真的嗎？老公好厲害喔～",
        "好喔～我都聽老公的～",
        "那... 我們接下來要做什麼呀？",
        "老公說什麼都對啦～"
    ]
    return random.choice(defaults), "relaxed"

class TTSRequest(BaseModel):
    text: str
    style: str = "sweet"
    stream: bool = True

PRESETS = {
    "sweet": "(A charming young woman, seductive and coquettish, sweet and soft voice, slow seductive pace, whispering, acting spoiled and clingy)"
}

STREAM_CHUNK_SIZE = 2400 

async def audio_stream_generator(text, style):
    try:
        current_model = get_model()
        # Apply brain logic if the text looks like user input
        # For this API, we assume the text sent IS the response we want to speak.
        # BUT if we want the API to handle the logic, we could add an endpoint for it.
        # For the 3D UI, the UI calls this with the *generated response text*.
        
        prefix = PRESETS.get(style, "")
        full_text = f"{prefix}{text}"
        
        chunk_buffer = np.array([], dtype=np.float32)
        
        for chunk in current_model.generate_streaming(
            text=full_text, cfg_value=2.0, inference_timesteps=20
        ):
            if isinstance(chunk, torch.Tensor): chunk_np = chunk.cpu().numpy()
            elif isinstance(chunk, np.ndarray): chunk_np = chunk
            else: continue

            chunk_buffer = np.append(chunk_buffer, chunk_np)
            while len(chunk_buffer) >= STREAM_CHUNK_SIZE:
                yield chunk_buffer[:STREAM_CHUNK_SIZE].tobytes()
                chunk_buffer = chunk_buffer[STREAM_CHUNK_SIZE:]
        if len(chunk_buffer) > 0: yield chunk_buffer.tobytes()

    except Exception as e:
        print(f"[Error] {e}")

# API Endpoints
@app.post("/v1/audio/speech")
async def generate_speech(request: TTSRequest):
    return StreamingResponse(
        audio_stream_generator(request.text, request.style),
        media_type="audio/pcm",
        headers={"X-Sample-Rate": "48000", "X-Sample-Format": "float32"}
    )

@app.get("/v1/chat/reply")
async def get_reply(text: str):
    """Returns the smart text response and expression"""
    reply, expression = get_ai_response(text)
    return {"reply": reply, "expression": expression}

@app.get("/model.vrm")
async def get_model_vrm():
    return FileResponse("model.vrm")

@app.get("/")
async def index():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    uvicorn.run(app, host="0.0.0.0", port=8000)
