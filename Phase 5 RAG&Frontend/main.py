from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_functionality import rag
import asyncio  # This is the function you've written above

app = FastAPI()


# Allow frontend (browser) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    question: str

@app.post("/chat")
async def chat(message: Message):
    response = rag(message.question)
    await asyncio.sleep(0.8)
    return {"answer": response}