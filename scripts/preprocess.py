import os.path
import numpy as np
import geopandas as gpd
import rasterio
from deepmap.data.ondemand import get_file
from deepmap.deepforest.utilities import shapefile_to_annotations
from deepmap.deepforest.preprocess import split_raster


THIS_DIR = os.path.dirname(__file__)
GDB = os.path.join(THIS_DIR, '../src/deepmap/data/kreike/KreikeSampleExtractedDataNam52022.gdb/')
TIFF = get_file('aerial_1970')
LAYER_NAME = 'Omuti1972'

SHAPEFILES_FOLDER = os.path.join(THIS_DIR, "preprocessed_data/shapefiles")
SPLIT_FOLDER = os.path.join(THIS_DIR, "preprocessed_data/split")


if __name__ == '__main__':

    os.makedirs(SHAPEFILES_FOLDER, exist_ok=True)
    os.makedirs(SPLIT_FOLDER, exist_ok=True)

    gdf = gpd.read_file(GDB, layer=LAYER_NAME)
    gdf["label"] = LAYER_NAME

    shapefile = os.path.join(SHAPEFILES_FOLDER, f"{LAYER_NAME}.shp")
    gdf.to_file(shapefile)

    annotations_df = shapefile_to_annotations(
        shapefile=shapefile,
        rgb=TIFF,
        buffer_size=0.15
    )

    with rasterio.open(TIFF) as src:
        numpy_image = src.read(1)
        numpy_image = np.dstack([numpy_image] * 3)  # Convert to 3 band

        df = split_raster(
            annotations_df,
            numpy_image=numpy_image,
            image_name=os.path.basename(TIFF),
            patch_size=1000,
            save_dir=SPLIT_FOLDER
        )

        random_state = np.random.RandomState()
        validation_frac = 0.2
        training_set = df.sample(frac=1-validation_frac, random_state=random_state)
        validation_set = df.sample(frac=validation_frac, random_state=random_state)

        training_set.to_csv(f"{SPLIT_FOLDER}/training.csv", index=None)
        validation_set.to_csv(f"{SPLIT_FOLDER}/validation.csv", index=None)
