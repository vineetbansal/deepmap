import os
import os.path
import geopandas as gpd
import numpy as np
from osgeo import gdal
from tqdm import tqdm
import slidingwindow
import pandas as pd
from deepmap.data.ondemand import get_file
from deepmap.deepforest.utilities import shapefile_to_annotations
from deepmap.deepforest.preprocess import select_annotations, save_crop

THIS_DIR = os.path.dirname(__file__)
GDB = os.path.join(THIS_DIR, '../src/deepmap/data/kreike/KreikeSampleExtractedDataNam52022.gdb/')
TIFF = get_file('aerial_1970')
LAYER_NAME = 'Omuti1972'
VALIDATION_FRAC = 0.2

SHAPEFILES_FOLDER = os.path.join(THIS_DIR, "preprocessed_data/shapefiles")
SPLIT_FOLDER = os.path.join(THIS_DIR, "preprocessed_data/split")


if __name__ == '__main__':

    os.makedirs(SHAPEFILES_FOLDER, exist_ok=True)
    os.makedirs(SPLIT_FOLDER, exist_ok=True)

    gdf = gpd.read_file(GDB, layer=LAYER_NAME)
    gdf["label"] = LAYER_NAME

    shapefile = os.path.join(SHAPEFILES_FOLDER, f"{LAYER_NAME}.shp")
    gdf.to_file(shapefile)

    annotations = shapefile_to_annotations(
        shapefile=shapefile,
        rgb=TIFF,
        buffer_size=0.15
    )

    dataset = gdal.Open(TIFF, gdal.GA_ReadOnly)

    xsize, ysize, nbands = dataset.RasterXSize, dataset.RasterYSize, dataset.RasterCount
    assert nbands == 1, "Can only deal with single band images"

    windows = slidingwindow.generateForSize(
        width=xsize,
        height=ysize,
        dimOrder=slidingwindow.DimOrder.HeightWidthChannel,
        maxWindowSize=1000,
        overlapPercent=0.05,
        transforms=[],
        overrideWidth=None,
        overrideHeight=None
    )

    all_crop_annotations = []
    band = dataset.GetRasterBand(1)
    for i, window in enumerate(tqdm(windows)):
        xoff, yoff, win_xsize, win_ysize = window.x, window.y, window.w, window.h
        crop_annotations = select_annotations(annotations, windows, i, False)

        if crop_annotations is not None:
            crop = band.ReadAsArray(xoff=xoff, yoff=yoff, win_xsize=win_xsize,
                                    win_ysize=win_ysize)
            all_crop_annotations.append(crop_annotations)
            save_crop(base_dir=SPLIT_FOLDER,
                      image_name=os.path.basename(TIFF), index=i,
                      crop=crop)

    all_crop_annotations = pd.concat(all_crop_annotations)
    all_crop_annotations.to_csv(
        os.path.join(SPLIT_FOLDER, "crop_annotations.csv"),
        index=False,
        header=True
    )

    random_state = np.random.RandomState()
    training_set = all_crop_annotations.sample(frac=1 - VALIDATION_FRAC,
                             random_state=random_state)
    validation_set = all_crop_annotations.sample(frac=VALIDATION_FRAC, random_state=random_state)

    training_set.to_csv(f"{SPLIT_FOLDER}/training.csv", index=None)
    validation_set.to_csv(f"{SPLIT_FOLDER}/validation.csv", index=None)
