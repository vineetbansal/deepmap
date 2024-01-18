import pandas as pd
import torch
from torch import optim
import pytorch_lightning as pl
import torchvision
from torchvision.models.detection.retinanet import RetinaNet
from deepmap.torch import dataset


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return tuple(zip(*batch))


def format_boxes(prediction, scores=True):
    df = pd.DataFrame(
        prediction["boxes"].cpu().detach().numpy(),
        columns=["xmin", "ymin", "xmax", "ymax"],
    )
    df["label"] = prediction["labels"].cpu().detach().numpy()

    if scores:
        df["score"] = prediction["scores"].cpu().detach().numpy()

    return df


class DeepmapModule(pl.LightningModule):
    def __init__(
        self,
        label_dict,
        transforms=None,
        training_file=None,
        validation_file=None,
    ):
        super().__init__()

        if torch.cuda.is_available():
            self.current_device = torch.device("cuda")
        else:
            self.current_device = torch.device("cpu")

        self.training_file = training_file
        self.validation_file = validation_file

        self.num_classes = len(label_dict)
        self.create_model()

        self.label_dict = label_dict
        self.numeric_to_label_dict = {v: k for k, v in label_dict.items()}

        # Add user supplied transforms
        if transforms is None:
            self.transforms = dataset.get_transform
        else:
            self.transforms = transforms

        self.save_hyperparameters()

    def create_model(self):
        resnet = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)

        model = RetinaNet(backbone=resnet.backbone, num_classes=self.num_classes)
        model.nms_thresh = 0.05
        model.score_thresh = 0.1

        self.model = model

    def create_trainer(self, max_epochs=10, logger=None, **kwargs):
        self.trainer = pl.Trainer(
            logger=logger,
            max_epochs=max_epochs,
            enable_checkpointing=True,
            accelerator="gpu",
            fast_dev_run=False,
            **kwargs,
        )

    def load_dataset(
        self, csv_file, augment=False, shuffle=True, batch_size=1
    ):
        return torch.utils.data.DataLoader(
            dataset.DeepMapDataset(
                csv_file=csv_file,
                label_dict=self.label_dict,
                transforms=self.transforms(augment=augment),
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=6,
        )

    def train_dataloader(self):
        return self.load_dataset(
            csv_file=self.training_file, augment=True, shuffle=True, batch_size=1
        )

    def val_dataloader(self):
        return self.load_dataset(
            csv_file=self.validation_file, augment=False, shuffle=False, batch_size=1
        )

    def predict_file(self, csv_file):
        self.model = self.model.to(self.current_device)
        self.model.eval()
        self.model.score_thresh = 0.1

        df = pd.read_csv(csv_file)
        # Dataloader (when not shuffled) returns a tensor for each image in order
        paths = df.image_path.unique()
        ds = dataset.DeepMapDataset(csv_file=csv_file, label_dict=self.label_dict,
                                    transforms=None, train=False)
        prediction_list = []
        with torch.no_grad():
            for i in ds:
                i = i.to(self.current_device)
                prediction = self.model(torch.unsqueeze(i, 0))
                prediction_list.append(prediction)

        prediction_list = [item for sublist in prediction_list for item in
                           sublist]

        results = []
        for index, prediction in enumerate(prediction_list):
            # If there is more than one class, apply NMS Loop through images and apply cross
            prediction = format_boxes(prediction)
            prediction["image_path"] = paths[index]
            results.append(prediction)

        results = pd.concat(results, ignore_index=True)

        results["label"] = results.label.apply(lambda x: self.numeric_to_label_dict[x])

        return results

    def training_step(self, batch, batch_idx):
        # Confirm model is in train mode
        self.model.train()

        # allow for empty data if data augmentation is generated
        path, images, targets = batch

        loss_dict = self.model.forward(images, targets)

        # sum of regression and classification loss
        losses = sum([loss for loss in loss_dict.values()])

        return losses

    def validation_step(self, batch, batch_idx):
        path, images, targets = batch

        self.model.train()
        with torch.no_grad():
            loss_dict = self.model.forward(images, targets)

        # sum of regression and classification loss
        losses = sum([loss for loss in loss_dict.values()])

        # Log loss
        for key, value in loss_dict.items():
            self.log("val_{}".format(key), value, on_epoch=True)

        return losses

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=10,
            verbose=True,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_classification",
        }
