import os
from dotenv import load_dotenv

from app.embeddings import get_embeddings
from app.llm import get_chain_context_huggingface, gptq, tokenizer

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

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
  print("question:", response["question"])
  print("answer:", response["answer"])

# ==================== ## ==================== # 

vectorstore = get_embeddings("app/embeddings")

# llm = ChatOpenAI()

models = {
  "indo_sentence": {
    "repo_id": "firqaaa/indo-sentence-bert-base",
    "local_dir": "./app/model/modules/indo-sentence-bert-base"
  },
  "gpt2": {
    "repo_id": "openai-community/gpt2",
    "local_dir": "./app/model/modules/gpt2"
  },
  "komodo": {
    "repo_id": "Yellow-AI-NLP/komodo-7b-base",
    "local_dir": "./app/model/modules/komodo"
  },
  "mistral": {
    "repo_id": "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ",
    "local_dir": "./app/model/modules/mistral"
  },
  "llama2": {
    "repo_id": "meta-llama/Llama-2-7b-chat-hf",
    "local_dir": "./app/model/modules/llama2"
  },
  "sealion": {
    "repo_id": "aisingapore/sea-lion-7b-instruct",
    "local_dir": "./app/model/modules/sealion"
  }
}
path = models["mistral"]["local_dir"]
token = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=path)

model_gptq = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=path,
    device_map="auto",
    torch_dtype=torch.float16
    )

llm = get_chain_context_huggingface(model_gptq, token)
# llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
# llm = AutoModelForCausalLM.from_pretrained(path)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# template = (
#     "Combine the chat history and follow up question into "
#     "a standalone question. Chat History: {chat_history}"
#     "Follow up question: {question}"
# )
# prompt = PromptTemplate.from_template(template)

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
)

def ask_question(question):
  response = conversation_chain({'question':question})
  print_response(response)

ask_question("apa itu AI")
ask_question("what is PKM KI")

# test chat history
ask_question("Apa isi pasal 2?")
ask_question("apakah ada pasal lain yang berkaitan dengan pasal tersebut?")