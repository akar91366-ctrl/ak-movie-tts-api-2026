
import asyncio
import io
import re
from typing import List

import edge_tts
import imageio_ffmpeg
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment

AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

app = FastAPI(
    title="AK Movie AI Studio - Burmese TTS API",
    description="Burmese text-to-speech API using edge-tts.",
    version="1.0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

VOICE_MAP = {
    "male": "my-MM-ThihaNeural",
    "female": "my-MM-NilarNeural",
}

PREVIEW_TEXT = {
    "male": "မင်္ဂလာပါ။ AK Movie AI Studio မှ ကြိုဆိုပါတယ်။ ကျွန်တော်က သီဟပါ။",
    "female": "မင်္ဂလာပါ။ AK Movie AI Studio မှ ကြိုဆိုပါတယ်။ ကျွန်မက နီလာပါ။",
}

MAX_TEXT_CHARS = 12000
CHUNK_SIZE = 250


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    gender: str = Field(default="female", pattern="^(male|female)$")


def chunk_text(text: str, max_chars: int = CHUNK_SIZE) -> List[str]:
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[။၊.!?\n])", text)
    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) <= max_chars:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = sentence
            continue

        if current:
            chunks.append(current)
            current = ""

        pieces = sentence.split()
        if len(pieces) > 1:
            piece = ""
            for word in pieces:
                candidate = f"{piece} {word}".strip()
                if len(candidate) <= max_chars:
                    piece = candidate
                else:
                    if piece:
                        chunks.append(piece)
                    piece = word
            if piece:
                current = piece
        else:
            for start in range(0, len(sentence), max_chars):
                part = sentence[start:start + max_chars]
                if start + max_chars < len(sentence):
                    chunks.append(part)
                else:
                    current = part

    if current:
        chunks.append(current)

    return [chunk for chunk in chunks if chunk]


async def synthesize_chunk(text: str, voice: str) -> bytes:
    communicator = edge_tts.Communicate(text, voice)
    data = bytearray()

    async for item in communicator.stream():
        if item.get("type") == "audio":
            data.extend(item["data"])

    if not data:
        raise RuntimeError("The TTS provider returned an empty audio chunk.")

    return bytes(data)


@app.get("/")
async def health_check():
    return {"status": "online", "service": "AK Movie AI Studio TTS API"}


@app.get("/api/voices")
async def get_voices():
    return {
        "voices": [
            {"gender": "male", "name": "Thiha (မြန်မာ အမျိုးသား)", "voice_code": VOICE_MAP["male"]},
            {"gender": "female", "name": "Nilar (မြန်မာ အမျိုးသမီး)", "voice_code": VOICE_MAP["female"]},
        ]
    }


@app.get("/api/preview")
async def preview_voice(gender: str = Query(default="female", pattern="^(male|female)$")):
    voice = VOICE_MAP[gender]

    try:
        audio = await synthesize_chunk(PREVIEW_TEXT[gender], voice)
        return Response(
            content=audio,
            media_type="audio/mpeg",
            headers={"Content-Disposition": f'inline; filename="preview_{gender}.mp3"'},
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"TTS provider error: {str(exc)[:300]}")


@app.post("/api/tts")
async def generate_tts(payload: TTSRequest):
    text = payload.text.strip()
    gender = payload.gender.lower()

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    chunks = chunk_text(text)

    if not chunks:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        combined = AudioSegment.empty()

        for chunk in chunks:
            mp3_bytes = await synthesize_chunk(chunk, VOICE_MAP[gender])
            segment = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
            combined += segment

        output = io.BytesIO()
        combined.export(output, format="mp3", bitrate="128k")

        return Response(
            content=output.getvalue(),
            media_type="audio/mpeg",
            headers={"Content-Disposition": 'attachment; filename="burmese_narration.mp3"'},
        )

    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"TTS generation failed: {str(exc)[:300]}")
