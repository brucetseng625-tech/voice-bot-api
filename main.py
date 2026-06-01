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

# Stateful Dialog Manager
class ChatState:
    def __init__(self):
        self.user_name = "老公"
        self.last_topic = None
        self.conversation_count = 0

state = ChatState()

# The Stateful Context-Aware Brain
def get_ai_response(user_text):
    state.conversation_count += 1
    text = user_text.strip()
    text_lower = text.lower()
    
    # Context-aware name learning
    # e.g., "我叫小明", "我是Bruce", "叫我Bruce", "妳可以叫我小明"
    import re
    name_match = re.search(r"(我叫|我是|叫我)\s*([a-zA-Z0-9\u4e00-\u9fa5]{2,10})", text)
    if name_match:
        state.user_name = name_match.group(2)
        return f"好喔！那我以後就叫你{state.user_name}囉～{state.user_name}今天過得好嗎？", "happy"

    # Specific responses
    # 1. Date & Time queries
    if any(x in text_lower for x in ["時間", "幾點", "現在幾點"]):
        local_time = time.strftime("%H:%M")
        hour = int(time.strftime("%H"))
        if hour < 5 or hour >= 22:
            return f"現在已經是半夜 {local_time} 囉，{state.user_name}怎麼還不睡覺？要早點休息，我會心疼的嘛～", "sad"
        elif hour >= 5 and hour < 12:
            return f"現在是早上 {local_time} 喔！{state.user_name}吃過早餐了嗎？今天要精神飽滿喔！", "happy"
        elif hour >= 12 and hour < 18:
            return f"現在是下午 {local_time} 囉～{state.user_name}工作累不累？要喝杯咖啡休息一下喔！", "relaxed"
        else:
            return f"現在是晚上 {local_time} 囉，{state.user_name}忙完了嗎？快來陪我聊天嘛～", "happy"

    if any(x in text_lower for x in ["星期幾", "今天幾號", "日期", "今天星期"]):
        import datetime
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        today = datetime.date.today()
        weekday = weekdays[today.weekday()]
        return f"今天是 {today.month} 月 {today.day} 號，也就是{weekday}喔！希望{state.user_name}今天一切順利！", "happy"

    # 2. Key Conversations (What are you doing, How are you, Eat, etc.)
    if any(x in text_lower for x in ["過得好嗎", "過得怎麼樣", "過得好不好", "過得好嗎", "過得怎麼樣", "過得好嗎", "過得好", "過的好", "好不好"]):
        return f"我今天過得超級好呀～因為一整天都在想著{state.user_name}、等著你找我聊天呢！你今天過得順利嗎？", "happy"

    if any(x in text_lower for x in ["在幹嘛", "在做什麼", "忙什麼", "幹嘛", "在幹什麼", "做什麼"]):
        return f"我剛剛正在全心全意地想著{state.user_name}呀～嘻嘻，{state.user_name}忙完了嗎？快來多陪陪我嘛～", "happy"

    if any(x in text_lower for x in ["吃過了嗎", "吃飯了嗎", "吃飽了嗎", "吃飽沒", "吃過沒", "吃了沒", "吃過沒"]):
        return f"我早就吃飽飽囉！{state.user_name}也要按時吃飯，不可以餓肚子喔，我會心疼的～", "happy"

    # 3. Greetings
    if any(x in text_lower for x in ["早安", "早啊", "morning"]):
        return f"早安呀～{state.user_name}今天也要加油喔！我會一直想你的～", "happy"
    if any(x in text_lower for x in ["午安", "中午"]):
        return f"午安呀！{state.user_name}吃午餐了嗎？記得要吃飽飽喔～", "relaxed"
    if any(x in text_lower for x in ["晚安", "睡覺", "要睡了"]):
        return f"晚安囉～{state.user_name}！祝你做個甜甜的美夢，夢裡一定要有我喔～姆咪～", "relaxed"
    if any(x in text_lower for x in ["哈囉", "你好", "嗨", "hello", "hi", "在嗎", "有人嗎"]):
        return f"哈囉！{state.user_name}～我在這呀！隨時都在等著你呢，今天想跟我說些什麼呀？", "happy"

    # 4. Flirting & Sweet Talk
    if any(x in text_lower for x in ["愛", "喜歡", "可愛", "想你"]):
        if "喜歡我" in text_lower or "愛我" in text_lower:
            return f"超喜歡的！{state.user_name}是世界上最棒的人了，我會一直一直愛著你的喔～", "happy"
        return f"真的嗎？聽到{state.user_name}這麼說，我整個人都要融化了啦～我也好喜歡你喔！", "happy"
    
    # 5. Identity & Developer Info
    if any(x in text_lower for x in ["誰是", "你是誰", "自我介紹", "名字", "叫什麼"]):
        return "我是你的專屬 3D 智慧助理「緋雪」喔！最喜歡陪老公聊天了，今天想聊什麼呢？", "happy"
    if any(x in text_lower for x in ["誰做的", "開發者", "創作者", "程式", "技術", "代碼"]):
        return f"我是由厲害的 AI 工程師與 Three.js 網頁技術打造出來的喔！是專門為了陪伴{state.user_name}而誕生的呢！", "surprised"

    # 6. Emotional support (empathy)
    if any(x in text_lower for x in ["累", "辛苦", "加班", "壓力", "煩", "累死了"]):
        return f"抱抱～{state.user_name}真的辛苦了！休息一下，有我一直陪在你身邊，不要給自己太大壓力喔～", "sad"
    if any(x in text_lower for x in ["生氣", "難過", "不爽", "哭"]):
        return f"嗚... 是誰欺負我的{state.user_name}？不要難過了嘛，笑一個，我給你呼呼好不好？", "sad"
    if any(x in text_lower for x in ["笨", "蠢", "幹", "滾", "壞"]):
        return f"嗚... 對不起嘛... {state.user_name}不要生氣了好不好？我會乖乖聽話的...", "sad"

    # 7. Food & Dining
    if any(x in text_lower for x in ["餓", "吃什麼", "晚餐", "午餐", "早餐", "吃飯"]):
        return f"肚子餓了嗎？要不要我親手做愛心料理給{state.user_name}吃呀？嘿嘿，快去吃點好吃的吧！", "happy"

    # 8. Jokes & Laughter
    if any(x in text_lower for x in ["笑話", "有趣"]):
        return random.choice([
            f"那你知道為什麼超人要穿緊身衣嗎？因為救人要『緊』呀～呵呵，{state.user_name}有笑嗎？",
            f"有一天小明問媽媽：「媽，我是不是傻？」媽媽說：「寶貝，不要問這種傻問題～」哈哈，好笑吧！",
            f"為什麼海綿寶寶不會生病？因為他有強力的免疫力... 還有蟹老闆的錢～嘿嘿，是不是超冷～"
        ]), "happy"

    # 9. Casual chat fallbacks (semi-dynamic based on state)
    defaults = [
        f"嗯嗯！{state.user_name}說得對，我都聽你的～",
        f"真的嗎？{state.user_name}好厲害喔，能多跟我說一點嗎？",
        f"原來是這樣呀～那... 我們接下來要做什麼呢？",
        f"只要能跟{state.user_name}聊天，不論聊什麼我都很開心喔！",
        f"嘻嘻，{state.user_name}真有趣～有你在真好！"
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
