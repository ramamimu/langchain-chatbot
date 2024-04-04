from app.embeddings import get_embeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

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

llm = ChatOpenAI()
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

template = (
    "Combine the chat history and follow up question into "
    "a standalone question. Chat History: {chat_history}"
    "Follow up question: {question}"
)
prompt = PromptTemplate.from_template(template)

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