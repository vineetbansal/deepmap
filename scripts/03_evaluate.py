import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import torch
from deepmap.torch.main import DeepmapModule


def plot_predictions(predictions, output_dir, ground_truth=None):
    os.makedirs(output_dir, exist_ok=True)
    for name, _predictions in predictions.groupby("image_path"):
        basename = os.path.splitext(os.path.basename(name))[0]
        image = np.array(Image.open(name).convert("RGB")).copy()

        # Plot predictions with blue boxes
        for _, row in _predictions.iterrows():
            image = cv2.rectangle(
                image,
                (int(row["xmin"]), int(row["ymin"])),
                (int(row["xmax"]), int(row["ymax"])),
                color=(255, 0, 0),  # BGR
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        # Plot ground truth with green boxes
        if ground_truth is not None:
            annotations = ground_truth[ground_truth.image_path == name]
            for _, row in annotations.iterrows():
                image = cv2.rectangle(
                    image,
                    (int(row["xmin"]), int(row["ymin"])),
                    (int(row["xmax"]), int(row["ymax"])),
                    color=(0, 255, 0),  # BGR
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

        cv2.imwrite(f"{output_dir}/{basename}.png", image)


if __name__ == "__main__":

    # TODO: We only do training/evaluation on a single feature for right now.
    feature_name = "Omuti1972"
    label_dict = {feature_name: 0}

    model = DeepmapModule(label_dict=label_dict)

    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    testing_file = f"scratch/{feature_name}/testing.csv"
    ground_truth = pd.read_csv(testing_file)
    predictions = model.predict_file(testing_file)
    plot_predictions(predictions, output_dir="predictions", ground_truth=ground_truth)
