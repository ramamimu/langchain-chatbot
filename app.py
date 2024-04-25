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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)