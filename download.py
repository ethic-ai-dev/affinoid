from huggingface_hub import snapshot_download, login

login()

snapshot_download(
    repo_id = "deepseek-ai/DeepSeek-V3.2-Speciale",
    local_dir = "/deepseek-model",
    local_dir_use_symlinks = False,
    resume_download = True
)
