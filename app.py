from fastapi import FastAPI
import uvicorn
from llm import conversation_chain
from pydantic import BaseModel

app = FastAPI()

@app.get("/ping")
async def root():
    return {"message": "pong"}

class Question(BaseModel):
   question: str

@app.post("/ask")
async def ask_question(item: Question):
  key = {'question': item.question}
  return {"answer": conversation_chain(key)["answer"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)