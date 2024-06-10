import time
from datasets import Dataset 
# import metrics
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate
import pandas as pd
import ast

def evaluate_result(context_path: str, file_to_save_path: str):
  print("================= START EVALUATE =================")
  print(f"calculating {context_path}")
  df_context = pd.read_csv(context_path)
  df_context['contexts'] = df_context['contexts'].apply(ast.literal_eval)


  data_sample = {
    'question': df_context['question'].to_list(),
    'answer': df_context['answer'].to_list(),
    'contexts': df_context['contexts'].to_list(),
    'ground_truth': df_context['ground_truth'].to_list(),
  }
  dataset = Dataset.from_dict(data_sample)

  result = evaluate(
      dataset,
      metrics=[
          context_precision,
          faithfulness,
          answer_relevancy,
          context_recall,
      ],
      raise_exceptions=False
  )

  # result
  df_evaluation_s1 = result.to_pandas()
  print(df_evaluation_s1.head())
  df_evaluation_s1.to_csv(file_to_save_path)
  print(f"success collecting {context_path}")
  print("================= FINISH EVALUATE =================")

list_path = [
  {
    "path": 'akademik',
    "is_finished": True
  },
  {
    "path": 'akademik s1',
    "is_finished": True
  },
  {
    "path": 'akademik s2',
    "is_finished": True
  },
  {
    "path": 'beasiswa',
    "is_finished": True
  },
  {
    "path": 'dana pendidikan',
    "is_finished": True
  },
  {
    "path": 'kerja praktik',
    "is_finished": False
  },
  {
    "path": 'magang',
    "is_finished": False
  },
  {
    "path": 'MBKM',
    "is_finished": False
  },
  {
    "path": 'program internasional',
    "is_finished": False
  },
  {
    "path": 'SKEM',
    "is_finished": False
  },
  {
    "path": 'tesis',
    "is_finished": False
  },
  {
    "path": 'wisuda',
    "is_finished": False
  },
  {
    "path": 'yudisium dan tugas akhir',
    "is_finished": False
  },
]

for i in list_path:
  if i['is_finished']: 
    continue

  # evaluate_result(f'dataset/{i["path"]}/evaluation context bert.csv', f'dataset/{i["path"]}/evaluation result bert.csv')
  evaluate_result(f'dataset/{i["path"]}/evaluation context openai.csv', f'dataset/{i["path"]}/evaluation result openai.csv')