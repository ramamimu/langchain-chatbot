from enum import Enum


class ModelName(Enum):
  INDO_SENTENCE="indo_sentence"
  GPT2="gpt2"
  KOMODO="komodo"
  MISTRAL="mistral"
  LLAMA2="llama2"
  SEALION2="sealion"
  MINILLM_V6="minilm-v6"
  MPNET_BASE2="mpnet-base-v2"
  MULTILINGUAL_MINILM="multilingual-minilm"
  LABSE="labse"
  NOMIC_EMBED="nomic-embed"
  MULTILINGUAL_E5_SMALL="multilingual-e5-small"

models = {
  ModelName.INDO_SENTENCE.value: {
    "repo_id": "firqaaa/indo-sentence-bert-base",
    "local_dir": "./app/model/modules/indo-sentence-bert-base"
  },
  # ModelName.GPT2.value: {
  #   "repo_id": "openai-community/gpt2",
  #   "local_dir": "./app/model/modules/gpt2"
  # },
  # ModelName.KOMODO.value: {
  #   "repo_id": "Yellow-AI-NLP/komodo-7b-base",
  #   "local_dir": "./app/model/modules/komodo"
  # },
  # ModelName.MISTRAL.value: {
  #   "repo_id": "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ",
  #   "local_dir": "./app/model/modules/mistral"
  # },
  # ModelName.LLAMA2.value: {
  #   "repo_id": "meta-llama/Llama-2-7b-chat-hf",
  #   "local_dir": "./app/model/modules/llama2"
  # },
  # ModelName.SEALION2.value: {
  #   "repo_id": "aisingapore/sea-lion-7b-instruct",
  #   "local_dir": "./app/model/modules/sealion"
  # },
  ModelName.MINILLM_V6.value: {
    "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
    "local_dir": "./app/model/modules/all-minilm"
  },
  ModelName.MPNET_BASE2.value: {
    "repo_id": "sentence-transformers/all-mpnet-base-v2",
    "local_dir": "./app/model/modules/mpnet-base-v2"
  },
  ModelName.MULTILINGUAL_MINILM.value: {
    "repo_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "local_dir": "./app/model/modules/multilingual-minilm"
  },
  ModelName.LABSE.value: {
    "repo_id": "sentence-transformers/LaBSE",
    "local_dir": "./app/model/modules/labse"
  },
  # ModelName.NOMIC_EMBED.value: {
  #   "repo_id": "nomic-ai/nomic-embed-text-v1.5",
  #   "local_dir": "./app/model/modules/nomic_embed"
  # },
  ModelName.MULTILINGUAL_E5_SMALL.value: {
    "repo_id": "intfloat/multilingual-e5-small",
    "local_dir": "./app/model/modules/multilingual-e5-small"
  }
}

# labse berat
# nomic failed to fulfill evaluation due to run out of GPU memory in IndonesianMongabayConservationClassification task