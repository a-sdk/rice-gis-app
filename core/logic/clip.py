"""
Clip operation module
"""

import os
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
from core.logic.utils import get_filename

# Fungsi untuk memotong citra bedasarkan shapefile poligon
def clip_raster(input_path, shp_path, output_folder, nodata_val=0):
    """
    Clip raster based on polygon shapefile.

    Parameters:
        input_path (str): Path to the input multiband raster file.
        shp_path (str): Path to the input polygon shapefile (.shp).
        output_folder (str): Directory where the output file will be saved.
        nodata_val (float, optional): Nodata value to set in the profile. Defaults to 0.
    Returns:
        str: Output path.
    """
    # Tentukan lokasi hasil clip
    base_name = get_filename(input_path)
    output_path = os.path.join(output_folder, f"{base_name}_clipped.tif")
    # Pastikan folder output ada, jika tidak, buat folder baru
    os.makedirs(output_folder, exist_ok=True)
    # Baca Shapefile Menggunakan GeoPandas
    # print("\nMemuat shapefile...")
    mask_gdf = gpd.read_file(shp_path)
    # print("Memuat citra...")
    with rio.open(input_path) as src:
        profile = src.profile
        geometries = mask_gdf.geometry
        # Melakukan masking
        # print("Memotong raster berdasarkan shapefile...")
        out_image, out_transform = mask(
            src,
            geometries, 
            crop=True,
            filled=True,
            nodata=nodata_val
            )
    # Perbarui metadata dengan informasi dari hasil clip
    profile.update(
        dtype="uint16",
        height=out_image.shape[1],
        width=out_image.shape[2],
        count=out_image.shape[0],
        transform=out_transform,
        BIGTIFF="YES",
        nodata=nodata_val,
        driver="GTiff"
    )
    # print("Menyimpan hasil clipping...")
    with rio.open(output_path, "w", **profile) as dest:
        dest.write(out_image)
    # print(f"File {output_filename} berhasil disimpan di {output_folder}")
    return output_path
    