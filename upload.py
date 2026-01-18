from huggingface_hub import login, upload_folder

login()

upload_folder(
    folder_path = "deepseek-model",
    repo_id = "joeldwaynepablos/Affine-Ultimate-HotkeyXYZ789",
    repo_type = "model"
)
