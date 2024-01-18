"""
Dataset model

https://pytorch.org/docs/stable/torchvision/models.html#object-detection-instance-segmentation-and-person-keypoint-detection

During training, the model expects both the input tensors, as well as a targets (list of dictionary), containing:

boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W

labels (Int64Tensor[N]): the class label for each ground-truth box

https://colab.research.google.com/github/benihime91/pytorch_retinanet/blob/master/demo.ipynb#scrollTo=0zNGhr6D7xGN

"""
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import albumentations
import albumentations.pytorch
import torch
from PIL import Image


def get_transform(augment):
    """Albumentations transformation of bounding boxs"""
    if augment:
        transform = albumentations.Compose(
            [albumentations.HorizontalFlip(p=0.5), albumentations.pytorch.ToTensorV2()],
            bbox_params=albumentations.BboxParams(
                format="pascal_voc", label_fields=["category_ids"]
            ),
        )

    else:
        transform = albumentations.Compose(
            [albumentations.pytorch.ToTensorV2()],
            bbox_params=albumentations.BboxParams(
                format="pascal_voc", label_fields=["category_ids"]
            ),
        )

    return transform


class DeepMapDataset(Dataset):
    def __init__(self, csv_file, label_dict, transforms=None, train=True):
        self.csv_file = csv_file
        self.annotations = pd.read_csv(csv_file)
        if transforms is None:
            self.transform = get_transform(augment=train)
        else:
            self.transform = transforms
        self.image_names = self.annotations.image_path.unique()
        self.label_dict = label_dict
        self.train = train
        self.image_converter = albumentations.Compose(
            [albumentations.pytorch.ToTensorV2()]
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        image = np.array(Image.open(img_name).convert("RGB")) / 255
        image = image.astype("float32")

        if self.train:
            # select annotations
            image_annotations = self.annotations[
                self.annotations.image_path == self.image_names[idx]
            ]
            targets = {}
            targets["boxes"] = image_annotations[
                ["xmin", "ymin", "xmax", "ymax"]
            ].values.astype(float)

            # Labels need to be encoded
            targets["labels"] = image_annotations.label.apply(
                lambda x: self.label_dict[x]
            ).values.astype(np.int64)

            # If image has no annotations, don't augment
            if np.sum(targets["boxes"]) == 0:
                boxes = boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.from_numpy(targets["labels"])
                # channels last
                image = np.rollaxis(image, 2, 0)
                image = torch.from_numpy(image)
                targets = {"boxes": boxes, "labels": labels}
                return self.image_names[idx], image, targets

            augmented = self.transform(
                image=image, bboxes=targets["boxes"], category_ids=targets["labels"]
            )
            image = augmented["image"]

            boxes = np.array(augmented["bboxes"])
            boxes = torch.from_numpy(boxes)
            labels = np.array(augmented["category_ids"])
            labels = torch.from_numpy(labels)
            targets = {"boxes": boxes, "labels": labels}

            return self.image_names[idx], image, targets

        else:
            # Mimic the train augmentation
            converted = self.image_converter(image=image)
            return converted["image"]
