'''
Extraction module
'''

# Libraries
import pandas as pd
import rasterio as rio
from rasterstats import zonal_stats
from core.logic.utils import get_filename
import geopandas as gpd
import os
import logging

logger = logging.getLogger(__name__)

# Fungsi untuk mengekstrak rata-rata nilai piksel dalam sub poligon
def extract_zonal_mean_pixels(shp_path, input_path, output_folder):
    """
    Extract the mean pixel values from a raster based on polygon zones defined in a shapefile.
    Parameters:
        shp_path (str): Path to the reference shapefile containing polygon geometries.
        input_path (str): Path to the raster feature stack file to be extracted from.
        output_folder (str): Directory where the output results will be saved.

    Returns:
        str: Path to the generated CSV file.
    """
    gdf = gpd.read_file(shp_path)

    # Menyiapkan kolom identitas
    # gdf["no_urut"] = gdf.groupby("id").cumcount() + 1
    gdf["Nama"] = "Poligon " + gdf["id"].astype(str) + " Komponen " + gdf["no_urut"].astype(str)
    
    # Mengambil koordinat X dan Y dari centroid poligon
    gdf["X"] = gdf.geometry.centroid.x
    gdf["Y"] = gdf.geometry.centroid.y

    # Menyiapkan DataFrame hasil dengan kolom koordinat awal
    hasil_ekstraksi = gdf[["id", "no_urut", "Nama", "X", "Y"]].copy() 

    with rio.open(input_path) as src:
        count = src.count
        if count > 7:
            nama_bands = [
                "RED", "GREEN", "BLUE", "M_GREEN", "M_RED", 
                "RED_EDGE", "NIR", "NDREI"
            ]
        else:
            logger.info("Raster biasa")
            nama_bands = [
                "RED", "GREEN", "BLUE", "M_GREEN", "M_RED", 
                "RED_EDGE", "NIR"
            ]
        for i, name in enumerate(nama_bands):
            logger.info(f"Extracting {name}")
            band_data = src.read(i + 1)
            
            stats = zonal_stats(
                vectors=gdf, 
                raster=band_data, 
                affine=src.transform, 
                stats=["mean"], 
                nodata=src.nodata,
                all_touched=True # Mengambil semua piksel yang bersentuhan
            )
            
            mean_values = [s.get("mean", src.nodata) for s in stats]
            hasil_ekstraksi[name] = mean_values

    # Cek baris yang mengandung NaN sebelum dihapus
    df_lengkap = pd.DataFrame(hasil_ekstraksi)
    # logger.info(df_lengkap.head)
    nan_rows = df_lengkap[df_lengkap.isna().any(axis=1)]
    
    if not nan_rows.empty:
        logger.info(f"PERHATIAN: Ditemukan {len(nan_rows)} komponen poligon dengan nilai NaN.")
        # logger.info(f"Poligon yang bermasalah: {nan_rows['Nama'].tolist()}")
    # ------------------------
    # Pembersihan dan pengurutan data
    # print(f"Membersihkan hasil ekstraksi...")
    df = pd.DataFrame(hasil_ekstraksi).dropna()
    df_urut = df.sort_values(by=["id", "no_urut"], ascending=True)
    df_urut = df_urut.drop(columns=["no_urut"])
    # df_urut.rename(columns={"no_urut": "id"}, inplace=True)
    
    # Simpan file
    os.makedirs(output_folder, exist_ok=True)
    base_name = get_filename(input_path)
    output_path = os.path.join(output_folder, f"{base_name}_pixel_val.csv")
    df_urut.to_csv(output_path, index=False)
    # print(f"Ekstraksi selesai...")
    # print(f"Total baris yang diekstrak: {df_urut.shape[0]}")
    # print(rf"File {output_filename} berhasil disimpan di {output_folder}")    
    return output_path 


        
             
