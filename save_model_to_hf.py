from huggingface_hub import HfApi

model_path = "outputs/checkpoint-46"
username = "Cherran"
MODEL_NAME = "medical_gemma3_1b_sft"
api = HfApi(token=  "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")

api.create_repo(
    repo_id = f"{username}/{MODEL_NAME}",
    repo_type="model"
)

api.upload_folder(
    repo_id = f"{username}/{MODEL_NAME}",
    folder_path = model_path
)
