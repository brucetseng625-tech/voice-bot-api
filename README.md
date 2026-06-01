
# 🗣️ Voice Bot API (One-Person Company Edition)

A local AI voice API server powered by **VoxCPM2** and **FastAPI**. 
Designed for the One-Person Company workflow: Zero cloud costs, 100% privacy, high-quality voice generation.

## Features
- **OpenAI-Compatible**: Mimics `/v1/audio/speech` structure.
- **Style Presets**: Switch between 'sweet' (default), 'professional', and 'casual' via API.
- **Apple Silicon Optimized**: Runs natively on MPS (Metal Performance Shaders).
- **Privacy First**: All generation happens locally. No data leaves your machine.

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Server**:
   ```bash
   python main.py
   ```
   Server will start at `http://localhost:8000`.

## API Usage

**POST /v1/audio/speech**

```json
{
  "text": "你好呀，今天工作辛苦啦！",
  "style": "sweet"
}
```
Returns a `.wav` audio file.
