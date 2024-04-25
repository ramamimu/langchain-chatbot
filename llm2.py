import os
from dotenv import load_dotenv

from app.embeddings import get_embeddings
from langchain_openai import ChatOpenAI

from huggingface_hub import login
import torch

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
login(HUGGINGFACEHUB_API_TOKEN)

# ============= helper function ============= # 
def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

def print_response(response):
  print("\n============ Content ============")
  print("RAW:", response)
  print("============")
  print("question:", response["question"])
  print("answer:", response["answer"])

# ==================== ## ==================== # 

vectorstore = get_embeddings("app/embeddings")
retriever=vectorstore.as_retriever(search_kwargs={"k": 1})

# ==================== ## ==================== # 
question="tahun berapa dokumen dibuat?"
doc=retriever.get_relevant_documents(question)

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Chain
chain = prompt | llm

# Run
# print(chain.invoke({"context": doc, "question": question}))

# ==================== message history ==================== # 
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableParallel

store={}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


chain = RunnableParallel({"output_message": ChatOpenAI()})
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

print(with_message_history.invoke({"input": question, "history": []},
    config={"configurable": {"session_id": "baz"}}))