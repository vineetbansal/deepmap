import os
from collections import namedtuple
from importlib.resources import files, as_file
import fiona
from sklearn.model_selection import train_test_split
import geopandas
import rasterio.mask
from PIL import Image
from deepmap.utils import convert_3D_2D
from tqdm import tqdm
from deepmap.utils import subset
from deepmap.data.ondemand import get_file


def extract_all_layers(input_tif_file, gdb_file, output_dir, padding=200):
    subset_tif = os.path.join(output_dir, "labelled_area.tif")
    subset(input_tif_file, gdb_file=gdb_file, output_tif=subset_tif)
    subset_raster = rasterio.open(subset_tif)

    Bbox = namedtuple(
        "Bbox", ["xmin", "ymin", "xmax", "ymax"]
    )  # pixel offsets from top left
    csv_header = "image_path,xmin,ymin,xmax,ymax,label\n"

    all_layers = fiona.listlayers(gdb_file)
    for class_id, class_name in enumerate(all_layers):
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(os.path.join(class_output_dir, "images"), exist_ok=True)

        f = open(f"{class_output_dir}/all.csv", "w")
        f.write(csv_header)

        gdf = geopandas.read_file(gdb_file, layer=class_name).to_crs(subset_raster.crs)

        gdf.geometry = convert_3D_2D(gdf.geometry)
        series = gdf["geometry"]  # GeoSeries

        with tqdm(total=len(series)) as pbar:
            pbar.set_description(f"Feature {class_name}")
            # For each shape found in the layer
            for i, shape in series.items():
                out_image, out_transform = rasterio.mask.mask(
                    subset_raster,
                    shape,
                    pad=True,
                    pad_width=padding,
                    crop=True,
                    filled=False,
                )
                im = Image.fromarray(out_image.squeeze(0))
                image_basename = f"{i:04d}"
                image_path = os.path.abspath(
                    f"{class_output_dir}/images/{image_basename}.png"
                )
                im.save(image_path)

                # Find the pixel width/height of the shape
                _minx, _miny, _maxx, _maxy = tuple(shape.bounds)
                _col_off1, _row_off1 = ~subset_raster.transform * (_minx, _miny)
                _col_off2, _row_off2 = ~subset_raster.transform * (_maxx, _maxy)
                _width, _height = _col_off2 - _col_off1, _row_off1 - _row_off2

                # By using rasterio.mask.mask with a padding, we know that the shape is centered w.r.t the image
                xmin = (im.width - _width) / 2.0
                xmax = (im.width + _width) / 2.0
                ymin = (im.height - _height) / 2.0
                ymax = (im.height + _height) / 2.0
                bbox = Bbox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

                f.write(
                    f"{image_path},{bbox.xmin},{bbox.ymin},{bbox.xmax},{bbox.ymax},{class_name}\n"
                )
                pbar.update(1)

        f.close()

        all_lines = open(f"{class_output_dir}/all.csv", "r").readlines()[
            1:
        ]  # skip header!
        # some classes have no labelled data; skip these
        if not all_lines:
            continue

        training, testing = train_test_split(all_lines, train_size=0.8)
        training, validation = train_test_split(training, train_size=0.9)
        with open(f"{class_output_dir}/training.csv", "w") as f:
            f.write(csv_header)
            f.writelines([f"{line}" for line in training])
        with open(f"{class_output_dir}/validation.csv", "w") as f:
            f.write(csv_header)
            f.writelines([f"{line}" for line in validation])
        with open(f"{class_output_dir}/testing.csv", "w") as f:
            f.write(csv_header)
            f.writelines([f"{line}" for line in testing])

    subset_raster.close()


if __name__ == "__main__":
    labeled_tif_file = get_file("aerial_1970")
    with as_file(
        files("deepmap.data").joinpath("kreike/KreikeSampleExtractedDataNam52022.gdb")
    ) as gdb_file:
        extract_all_layers(
            input_tif_file=labeled_tif_file, gdb_file=gdb_file, output_dir="scratch"
        )
