"""FastAPI + OpenRouter example.

WildEdge is injected via `wildedge run` before the app starts, so inference
tracking happens automatically for every chat.completions.create call.

Run with: see demo.sh
Requires: WILDEDGE_DSN and OPENROUTER_API_KEY environment variables.
"""

import os
import pathlib

from fastapi import FastAPI
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel

STATIC = pathlib.Path(__file__).parent / "static"

app = FastAPI()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)


class ChatRequest(BaseModel):
    prompt: str
    model: str = "openai/gpt-4o-mini"


class ChatResponse(BaseModel):
    response: str
    model: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    completion = client.chat.completions.create(
        model=req.model,
        messages=[{"role": "user", "content": req.prompt}],
        max_tokens=512,
    )
    return ChatResponse(
        response=completion.choices[0].message.content,
        model=completion.model,
    )
