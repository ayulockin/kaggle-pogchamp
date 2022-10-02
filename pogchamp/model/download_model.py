import wandb


def download_model(artifact_url):
    wandb_api = wandb.Api()
    artifact = wandb_api.artifact(artifact_url, type="model")
    artifact_dir = artifact.download()

    return artifact_dir
