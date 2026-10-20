"""
Masking operation module
"""

import os
import numpy as np
from core.logic.utils import get_filename
import rasterio as rio
import logging

logger = logging.getLogger(__name__)
# Fungsi untuk melakukan masking pada tumpukan fitur
def mask_stack(input_path, mask_path, output_folder, logic=lambda x: x==1, nodata_val=0):
    """
    Apply a boolean mask to a stack (raster) based on a specified condition.
    Parameters:
        input_path (str): Path to the input raster stack file.
        mask_path (str): Path to the mask raster file.
        output_folder (str): Directory where the output file will be saved.
        logika (callable, optional): Condition function (lambda) used for masking. Defaults to lambda x: x == 1.
        nodata_val (float, optional): Nodata value to set in the profile. Defaults to 0.
    Returns:
        str: Path to the generated masked output file.
    """
    with rio.open(mask_path) as src_mask:
            mask_data = src_mask.read(1)
    # Membuat boolean mask
    mask_valid = logic(mask_data)
    # Mengecualikan nilai nodata
    mask_valid[mask_data == nodata_val] = False           
    logger.info(f"Boolean mask generatid. Total valid pixel: {np.sum(mask_valid)}")
    # print(f"Membuka tumpukan fitur...")
    os.makedirs(output_folder, exist_ok=True)
    base_name = get_filename(input_path)
    output_path = os.path.join(output_folder, f"{base_name}_mask result.tif")
    with rio.open(input_path) as src_data:
        # Membaca fitur sekaligus
        data_stack = src_data.read()
        profile = src_data.profile
        # print("Menerapkan mask ke semua fitur...")
        data_stack[:, ~mask_valid] = nodata_val
        
        # Perbarui profile untuk file output agar konsisten
        profile.update(
            dtype="float32",
            count=data_stack.shape[0], 
            nodata=nodata_val
        )
        # print("Menyimpan hasil masking...")
        with rio.open(output_path, "w", **profile) as dest:
            dest.write(data_stack)
    # print(f"File {nf}_masked.tif berhasil disimpan di {output_folder}")
    return output_path

def water_pred_mask(input_path, ndvi_path, output_folder, nodata_val=0):
    """
    Apply a specialized masking method for water adequacy 
    prediction based on NDVI and morphological operations.
    Parameters:
        input_path (str): Path to the input multiband raster file.
        ndvi_path (str): Path to NDVI raster file.
        output_folder (str): Directory where the output file will be saved.
        nodata_val (float, optional): Nodata value to set in the profile. Defaults to 0.
    Returns:
        str: Path to the generated masked output file.
    """

    from skimage import exposure
    from skimage.filters import threshold_local, gaussian
    from skimage.morphology import remove_small_objects, remove_small_holes, closing, disk
    from scipy import ndimage

    os.makedirs(output_folder, exist_ok=True)
    base_name = get_filename(input_path)
    output_path = os.path.join(output_folder, f"{base_name}_water mask.tif")

    with (
         rio.open(ndvi_path) as src_mask, 
         rio.open(input_path) as src_img
        ):
            ndvi = src_mask.read(1)
            img = src_img.read()
            profile = src_img.profile.copy()

    ndvi_enhanced = exposure.equalize_adapthist(ndvi, kernel_size=32, clip_limit=0.02)
    ndvi_blur = gaussian(ndvi_enhanced, sigma=0.5)
    local_thresh = threshold_local(ndvi_blur, block_size=101, offset=0.01, method='gaussian')
    mask_binary = (ndvi_blur > local_thresh) & (ndvi > 0.1)
    
    mask_cleaned = closing(mask_binary, disk(1))
    mask_cleaned = remove_small_objects(mask_cleaned, max_size=100)
    mask_cleaned = remove_small_holes(mask_cleaned, max_size=500)
    mask_cleaned = ndimage.median_filter(mask_cleaned, size=3)
    masked_img = img.copy()  
    masked_img[:, ~mask_cleaned] = nodata_val  

    profile.update(
        dtype=img.dtype,
        count=src_img.count, 
        nodata=nodata_val
    )

    with rio.open(output_path, "w", **profile) as dest: 
         dest.write(masked_img)

    return output_path