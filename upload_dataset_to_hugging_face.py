from huggingface_hub import HfApi, HfFolder, Repository, create_repo

# (1) Créer un repo (public ou privé)
create_repo(repo_id="roberta-fine-tuned-amazon", private=True)

# (2) Push le dossier du modèle
from huggingface_hub import upload_folder

upload_folder(
    repo_id="inverate/roberta-fine-tuned-amazon",
    folder_path="fine_tuned_roberta",
    path_in_repo=".",
)
