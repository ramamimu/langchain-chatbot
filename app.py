from fastapi import FastAPI
import uvicorn
from llm import conversation_chain, create_chain
from pydantic import BaseModel

app = FastAPI()

@app.get("/ping")
async def root():
    return {"message": "pong"}

class Question(BaseModel):
   question: str
   sid: str

chain_store = {
   "123a": create_chain(),
   "123b": create_chain()
}

@app.post("/ask")
async def ask_question(item: Question):
    key = {'question': item.question}
    chain = chain_store[item.sid]
    return chain(key)

import random
from lorem_text import lorem
import asyncio
from typing import List
from sse_starlette.sse import EventSourceResponse

async def lorem_generator():
  randomizer = random.randint(1, 5)
  lorem_text:str = lorem.paragraphs(randomizer)
  splitted_lorem:List[str] = lorem_text.split(' ')
  
  new_text = ""
  for i in splitted_lorem:
    new_text += f"{i} "
    yield f"{i} "
    await asyncio.sleep(0.1)

@app.post("/ask/stream")
async def generate_stream(item: Question):
    return EventSourceResponse(lorem_generator(), media_type='text/event-stream')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)