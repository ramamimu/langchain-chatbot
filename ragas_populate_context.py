from enum import Enum
import pandas as pd

# get embedding
# load all local embedded texts
from langchain_community.vectorstores import FAISS
from app.model import load_model
from langchain.embeddings.openai import OpenAIEmbeddings

class EmbeddingModel(Enum):
  OPENAI = 'openai'
  INDO_BERT_SENTENCE = 'indo-bert-sentence'

def get_embedding_model(model: str = 'openai'):
  if model == 'openai':
    embedding_model = OpenAIEmbeddings()
    fvs_init_path="./app/embeddings/openai"
    return embedding_model, fvs_init_path
  
  embedding_model = load_model("./app/model/modules/indo-sentence-bert-base")
  fvs_init_path="./app/embeddings/bert"
  return embedding_model, fvs_init_path


def get_vectorstore(embedding_model, fvs_init_path, df_dataset):
  faiss = None

  for _, row in df_dataset.iterrows():
    title = row['Judul']
    faiss = FAISS.load_local(folder_path=f"{fvs_init_path}/{title}", embeddings=embedding_model, allow_dangerous_deserialization=True)

  return faiss

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# prompt
template = """SYSTEM: Anda adalah chatbot interaktif bernama TANYABOT, jawablah pertanyaan secara detail dari konteks yang diberikan. Jika Anda tidak mengetahui jawabannya, katakan "saya tidak tahu".
  jawablah pertanyaan menggunakan Bahasa Indonsia.
  CONTEXT: {context}

  Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Chain
# use case
# chain.invoke({"context": doc, "question": "tahun berapa dokumen dibuat?"})
from langchain_core.output_parsers import StrOutputParser
chain = prompt | llm | StrOutputParser()

# get question and answer from dataframe
def get_question_ground_truth_from_dataset(df_dataset):
  questions = []
  expected_answers = []

  for _, row in df_dataset.iterrows():
    question = row['Question']
    expected_answer = row['Expected Answer']

    questions.append(question)
    expected_answers.append(expected_answer)
  
  return questions, expected_answers

# get context and generated answer from dataframe
def get_context_answer(questions, vestorstore, k=3):
  print("========= CHAIN CALLED =========")
  all_context = []
  all_answer = []
  for question in questions:
    contexts = []

    vs_result = vestorstore.similarity_search(question, kwargs={"k": k})
    for item in vs_result:
      contexts.append(item.page_content)

    all_context.append(contexts)
    all_answer.append(chain.invoke({"context": vs_result, "question": question}))
  return all_context, all_answer

from datasets import Dataset 
# import metrics
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate
def evaluate(questions, ground_truth, contexts, answers, file_to_save_path):
  print("================= START EVALUATE =================")
  data_samples_s1 = {
    'question': questions,
    "ground_truth": ground_truth,
    "contexts": contexts,
    "answer": answers
  }

  dataset = Dataset.from_dict(data_samples_s1)

  result = evaluate(
      dataset,
      metrics=[
          context_precision,
          faithfulness,
          answer_relevancy,
          context_recall,
      ],
  )

  # result
  df_evaluation_s1 = result.to_pandas()
  print(df_evaluation_s1.head())
  df_evaluation_s1.to_csv(file_to_save_path)

def calculate_evaluation(df_dataset_path, df_qa_path, embedding_model_name, save_local_path):
  df_dataset = pd.read_csv(df_dataset_path)
  df_qa = pd.read_csv(df_qa_path)
  embedding_model, model_path = get_embedding_model(embedding_model_name)
  vectorstore = get_vectorstore(embedding_model, model_path, df_dataset)
  questions, expected_answers = get_question_ground_truth_from_dataset(df_qa)
  contexts, answers = get_context_answer(questions, vectorstore)
  df_contexts = pd.DataFrame({
    'question': questions,
    "ground_truth": expected_answers,
    "contexts": contexts,
    "answer": answers
  })
  df_contexts.to_csv(save_local_path)
  print(f"{df_dataset_path} saved into {save_local_path}")
  # evaluate(questions, expected_answers, contexts, answers, result_evaluation_path)


# -------------------- EXAMPLE TEST Kerja Praktik File -------------------- #
# calculate_evaluation(
#   'dataset/Dataset IF-Tegration - Kerja Praktik.csv',
#   'dataset/Question Evaluation - File KP QA.csv',
#   EmbeddingModel.OPENAI.value,
#   'evaluation/evaluation kp dataset.csv'
# )

# -------------------- Akademik S1
# calculate_evaluation('dataset/akademik s1/Dataset IF-Tegration - Akademik S1.csv',
#                      'dataset/akademik s1/QA Generator - Akademik S1.csv',
#                      EmbeddingModel.OPENAI.value,
#                      'dataset/akademik s1/evaluation context openai.csv')

# calculate_evaluation('dataset/akademik s1/Dataset IF-Tegration - Akademik S1.csv',
#                      'dataset/akademik s1/QA Generator - Akademik S1.csv',
#                      EmbeddingModel.INDO_BERT_SENTENCE.value,
#                      'dataset/akademik s1/evaluation context bert.csv')

# -------------------- Akademik S2
# calculate_evaluation('dataset/akademik s2/Dataset IF-Tegration - Akademik S2.csv',
#                      'dataset/akademik s2/QA Generator - Akademik S2.csv',
#                      EmbeddingModel.INDO_BERT_SENTENCE.value,
#                      'dataset/akademik s2/evaluation context bert.csv')

# calculate_evaluation('dataset/akademik s2/Dataset IF-Tegration - Akademik S2.csv',
#                      'dataset/akademik s2/QA Generator - Akademik S2.csv',
#                      EmbeddingModel.OPENAI.value,
#                      'dataset/akademik s2/evaluation context openai.csv')

# -------------------- kerja praktik
# calculate_evaluation('dataset/kerja praktik/Dataset IF-Tegration - kerja praktik.csv',
#                      'dataset/kerja praktik/QA Generator - kerja praktik.csv',
#                      EmbeddingModel.INDO_BERT_SENTENCE.value,
#                      'dataset/kerja praktik/evaluation context bert.csv')

calculate_evaluation('dataset/kerja praktik/Dataset IF-Tegration - kerja praktik.csv',
                     'dataset/kerja praktik/QA Generator - kerja praktik.csv',
                     EmbeddingModel.OPENAI.value,
                     'dataset/kerja praktik/evaluation context openai.csv')

# -------------------- magang
# calculate_evaluation('dataset/magang/Dataset IF-Tegration - magang.csv',
#                      'dataset/magang/QA Generator - magang.csv',
#                      EmbeddingModel.INDO_BERT_SENTENCE.value,
#                      'dataset/magang/evaluation context bert.csv')

# calculate_evaluation('dataset/magang/Dataset IF-Tegration - magang.csv',
#                      'dataset/magang/QA Generator - magang.csv',
#                      EmbeddingModel.OPENAI.value,
#                      'dataset/magang/evaluation context openai.csv')

# -------------------- tesis
# calculate_evaluation('dataset/tesis/Dataset IF-Tegration - tesis.csv',
#                      'dataset/tesis/QA Generator - tesis.csv',
#                      EmbeddingModel.INDO_BERT_SENTENCE.value,
#                      'dataset/tesis/evaluation context bert.csv')

# calculate_evaluation('dataset/tesis/Dataset IF-Tegration - tesis.csv',
#                      'dataset/tesis/QA Generator - tesis.csv',
#                      EmbeddingModel.OPENAI.value,
#                      'dataset/tesis/evaluation context openai.csv')

# -------------------- MBKM
# calculate_evaluation('dataset/MBKM/Dataset IF-Tegration - MBKM.csv',
#                      'dataset/MBKM/QA Generator - MBKM.csv',
#                      EmbeddingModel.INDO_BERT_SENTENCE.value,
#                      'dataset/MBKM/evaluation context bert.csv')

calculate_evaluation('dataset/MBKM/Dataset IF-Tegration - MBKM.csv',
                     'dataset/MBKM/QA Generator - MBKM.csv',
                     EmbeddingModel.OPENAI.value,
                     'dataset/MBKM/evaluation context openai.csv')

# -------------------- beasiswa
# calculate_evaluation('dataset/beasiswa/Dataset IF-Tegration - beasiswa.csv',
#                      'dataset/beasiswa/QA Generator - beasiswa.csv',
#                      EmbeddingModel.INDO_BERT_SENTENCE.value,
#                      'dataset/beasiswa/evaluation context bert.csv')

calculate_evaluation('dataset/beasiswa/Dataset IF-Tegration - beasiswa.csv',
                     'dataset/beasiswa/QA Generator - beasiswa.csv',
                     EmbeddingModel.OPENAI.value,
                     'dataset/beasiswa/evaluation context openai.csv')

# -------------------- program internasional
# calculate_evaluation('dataset/program internasional/Dataset IF-Tegration - program internasional.csv',
#                      'dataset/program internasional/QA Generator - program internasional.csv',
#                      EmbeddingModel.INDO_BERT_SENTENCE.value,
#                      'dataset/program internasional/evaluation context bert.csv')

calculate_evaluation('dataset/program internasional/Dataset IF-Tegration - program internasional.csv',
                     'dataset/program internasional/QA Generator - program internasional.csv',
                     EmbeddingModel.OPENAI.value,
                     'dataset/program internasional/evaluation context openai.csv')

# -------------------- akademik
# calculate_evaluation('dataset/akademik/Dataset IF-Tegration - akademik.csv',
#                      'dataset/akademik/QA Generator - akademik.csv',
#                      EmbeddingModel.INDO_BERT_SENTENCE.value,
#                      'dataset/akademik/evaluation context bert.csv')

calculate_evaluation('dataset/akademik/Dataset IF-Tegration - akademik.csv',
                     'dataset/akademik/QA Generator - akademik.csv',
                     EmbeddingModel.OPENAI.value,
                     'dataset/akademik/evaluation context openai.csv')

# -------------------- yudisium dan tugas akhir
# calculate_evaluation('dataset/yudisium dan tugas akhir/Dataset IF-Tegration - yudisium dan tugas akhir.csv',
#                      'dataset/yudisium dan tugas akhir/QA Generator - yudisium dan tugas akhir.csv',
#                      EmbeddingModel.INDO_BERT_SENTENCE.value,
#                      'dataset/yudisium dan tugas akhir/evaluation context bert.csv')

calculate_evaluation('dataset/yudisium dan tugas akhir/Dataset IF-Tegration - yudisium dan tugas akhir.csv',
                     'dataset/yudisium dan tugas akhir/QA Generator - yudisium dan tugas akhir.csv',
                     EmbeddingModel.OPENAI.value,
                     'dataset/yudisium dan tugas akhir/evaluation context openai.csv')

# -------------------- wisuda
# calculate_evaluation('dataset/wisuda/Dataset IF-Tegration - wisuda.csv',
#                      'dataset/wisuda/QA Generator - wisuda.csv',
#                      EmbeddingModel.INDO_BERT_SENTENCE.value,
#                      'dataset/wisuda/evaluation context bert.csv')

calculate_evaluation('dataset/wisuda/Dataset IF-Tegration - wisuda.csv',
                     'dataset/wisuda/QA Generator - wisuda.csv',
                     EmbeddingModel.OPENAI.value,
                     'dataset/wisuda/evaluation context openai.csv')

# -------------------- SKEM
# calculate_evaluation('dataset/SKEM/Dataset IF-Tegration - SKEM.csv',
#                      'dataset/SKEM/QA Generator - SKEM.csv',
#                      EmbeddingModel.INDO_BERT_SENTENCE.value,
#                      'dataset/SKEM/evaluation context bert.csv')

calculate_evaluation('dataset/SKEM/Dataset IF-Tegration - SKEM.csv',
                     'dataset/SKEM/QA Generator - SKEM.csv',
                     EmbeddingModel.OPENAI.value,
                     'dataset/SKEM/evaluation context openai.csv')

# -------------------- dana pendidikan
# calculate_evaluation('dataset/dana pendidikan/Dataset IF-Tegration - dana pendidikan.csv',
#                      'dataset/dana pendidikan/QA Generator - dana pendidikan.csv',
#                      EmbeddingModel.INDO_BERT_SENTENCE.value,
#                      'dataset/dana pendidikan/evaluation context bert.csv')

calculate_evaluation('dataset/dana pendidikan/Dataset IF-Tegration - dana pendidikan.csv',
                     'dataset/dana pendidikan/QA Generator - dana pendidikan.csv',
                     EmbeddingModel.OPENAI.value,
                     'dataset/dana pendidikan/evaluation context openai.csv')
