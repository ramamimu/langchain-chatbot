import os 

from huggingface_hub import snapshot_download, login

login(os.getenv("HUGGINGFACEHUB_API_TOKEN"))

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

snapshot_download(repo_id=models["sealion"]["repo_id"],
                   local_dir=models["sealion"]["local_dir"])
