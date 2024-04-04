from huggingface_hub import snapshot_download
snapshot_download(repo_id="firqaaa/indo-sentence-bert-base",
                   local_dir="./app/model/modules/indo-sentence-bert-base")
