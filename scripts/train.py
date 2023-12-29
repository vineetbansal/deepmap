import os.path
import pandas as pd
from deepmap.deepforest import main
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


THIS_DIR = os.path.dirname(__file__)
SPLIT_FOLDER = os.path.join(THIS_DIR, "preprocessed_data/split")
CONFIG_FILE = os.path.join(THIS_DIR, "config.yml")

if __name__ == '__main__':

    training_csv = f"{SPLIT_FOLDER}/training.csv"
    validation_csv = f"{SPLIT_FOLDER}/validation.csv"

    all_annotations = pd.concat([pd.read_csv(training_csv), pd.read_csv(validation_csv)])
    label_dict = {label: i for i, label in enumerate(all_annotations['label'].unique())}

    model = main.deepforest(config_file=CONFIG_FILE, label_dict=label_dict)
    model.config["train"]["csv_file"] = training_csv
    model.config["train"]["root_dir"] = os.path.dirname(training_csv)
    model.config["validation"]["csv_file"] = validation_csv
    model.config["validation"]["root_dir"] = os.path.dirname(validation_csv)

    callback = ModelCheckpoint(dirpath='checkpoints',
                               monitor='box_recall',
                               mode="max",
                               save_top_k=3,
                               filename="box_recall-{epoch:02d}-{box_recall:.2f}")

    model.create_trainer(
        logger=TensorBoardLogger(save_dir='.'),
        callbacks=[callback]
    )
    model.trainer.fit(model)