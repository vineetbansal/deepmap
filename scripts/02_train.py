import torch
from pytorch_lightning.loggers import TensorBoardLogger
from deepmap.torch.main import DeepmapModule
from deepmap.data.ondemand import get_file


if __name__ == "__main__":

    # TODO: We only do training/evaluation on a single feature for right now.
    feature_name = "Omuti1972"
    label_dict = {feature_name: 0}

    model = DeepmapModule(
        training_file=f"scratch/{feature_name}/training.csv",
        validation_file=f"scratch/{feature_name}/validation.csv",
        label_dict=label_dict,
    )

    # Model trained on data from the National Ecological Observatory Network (NEON)
    # See https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13472
    state_dict = torch.load(get_file('neon'), map_location=model.device)
    model.model.load_state_dict(state_dict)

    model.create_trainer(
        max_epochs=10,
        logger=TensorBoardLogger(save_dir="lightning_logs/"),
        log_every_n_steps=1,
    )
    model.trainer.fit(model)

    torch.save(model.state_dict(), "model.pth")