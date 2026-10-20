"""
Transformation and segmentation module
"""

import numpy as np
import pandas as pd
import rasterio as rio
from core.logic.utils import otsu_threshold, save_raster, get_filename
from core.logic.modul_klasifikasi import separate_weeds
from path_config import AppPaths
import logging
import os

logger = logging.getLogger(__name__)

def safe_div(num, denom):
    """
    Perform safe element-wise division, replacing NaN and infinite results with zero.
    Parameters:
        num (float or np.ndarray): Numerator.
        denom (float or np.ndarray): Denominator.
    Returns:
        float or np.ndarray: Computed division result with NaNs and infinities replaced by 0.
    """ 
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.true_divide(num, denom)
    res = np.where(np.isnan(res) | np.isinf(res), 0.0, res)
    return res

def trim_norm_df(df, low_per=10, high_per=90):
    """
    Trim extreme values of a DataFrame based on percentiles (ignoring zeros and NaNs) 
    and normalize the results (scaling by 65535.0).
    Parameters:
        df (pd.DataFrame): DataFrame containing the numerical data to process.
        low_per (float, optional): Lower percentile threshold. Defaults to 10.
        high_per (float, optional): Upper percentile threshold. Defaults to 90.
    Returns:
        pd.DataFrame: Trimmed and normalized DataFrame.
    """
    data_array = df.values.astype(float)
    # Filter data untuk mencari ambang batas
    valid_mask = (data_array > 0) & (~np.isnan(data_array))
    valid_data = data_array[valid_mask]

    if valid_data.size == 0:
        return df 
    
    low_val, high_val = np.percentile(valid_data, [low_per, high_per])
    
    # Proses Trimming & Normalisasi
    trimmed_array = np.clip(data_array, low_val, high_val)
    normalized_array = trimmed_array / 65535.0
    return pd.DataFrame(normalized_array, columns=df.columns)

def calculate_savi(nir, m_red, L):
    """
    Calculate the Soil-Adjusted Vegetation Index (SAVI) for remote sensing imagery.
    Parameters:
        nir (np.ndarray): NumPy array representing the Near-Infrared (NIR) band.
        m_red (np.ndarray): NumPy array representing the Multispectral Red band.
        L (float): Soil brightness correction factor (typically ranges from 0 to 1).
    Returns:
        np.ndarray: Computed SAVI array clipped between -1.0 and 2.0.
    """

    nir_f = nir.astype(np.float32)
    m_red_f = m_red.astype(np.float32)
    num = nir_f - m_red_f 
    denom = nir_f + m_red_f + L
    res = safe_div(num, denom)
    savi = res * (1.0 + L)
    savi = np.clip(savi, -1.0, 2)
    return savi

def calculate_ndvi(nir, m_red):
    """
    Calculate the Normalize Difference Vegetation Index (NDVI) for remote sensing imagery.
    Parameters:
        nir (np.ndarray): NumPy array representing the Near-Infrared (NIR) band.
        m_red (np.ndarray): NumPy array representing the Multispectral Red band.
    Returns:
        np.ndarray: Computed NDVI array clipped between -1.0 and 1.
    """
    nir_f = nir.astype(np.float32)
    m_red_f = m_red.astype(np.float32)
    num = nir_f - m_red_f
    denom = nir_f + m_red_f
    ndvi = safe_div(num, denom)
    ndvi = np.clip(ndvi, -1.0, 1.0)
    return ndvi

def calculate_gndvi(nir, m_green):
    """
    Calculate the Green Normalize Difference Vegetation Index (GNDVI) for remote sensing imagery.
    Parameters:
        nir (np.ndarray): NumPy array representing the Near-Infrared (NIR) band.
        m_green (np.ndarray): NumPy array representing the Multispectral Green band.
    Returns:
        np.ndarray: Computed GNDVI array clipped between -1.0 and 1.
    """
    nir_f = nir.astype(np.float32)
    m_green_f = m_green.astype(np.float32)
    num = nir_f - m_green_f
    denom = nir_f + m_green_f
    gndvi = safe_div(num, denom)
    gndvi = np.clip(gndvi, -1.0, 1.0)
    return gndvi

def calculate_ndrei(nir, red_edge):
    """
    Calculate the Normalize Difference Red Edge Index (NDREI) for remote sensing imagery.
    Parameters:
        nir (np.ndarray): NumPy array representing the Near-Infrared (NIR) band.
        red_edge (np.ndarray): NumPy array representing the Red Edge band.
    Returns:
        np.ndarray: Computed NDREI array clipped between -1.0 and 1.
    """
    nir_f = nir.astype(np.float32)
    re_f = red_edge.astype(np.float32)
    num = nir_f - re_f 
    denom = nir_f + re_f 
    ndre = safe_div(num, denom)
    ndre = np.clip(ndre, -1.0, 1.0)
    return ndre

def calculate_evi(nir, m_red, blue):
    """
    Calculate the Enhanced Vegetation Index (EVI) for remote sensing imagery.
    Parameters:
        nir (np.ndarray): NumPy array representing the Near-Infrared (NIR) band.
        m_red (np.ndarray): NumPy array representing the Multispectral Red band.
        blue (np.ndarray): NumPy array representing the Visible Blue band.
    Returns:
        np.ndarray: Computed EVI array clipped between -1.0 and 1.
    """
    nir_f = nir.astype(np.float32)
    m_red_f = m_red.astype(np.float32)
    blue_f = blue.astype(np.float32)
    num = nir_f - m_red_f
    denom = nir_f + 6 * m_red_f - 7.5 * blue_f + 1
    evi = np.clip(2.5 * safe_div(num, denom), -1.0, 1.0)
    return evi

def calculate_vidvi(red, green, blue):
    """
    Calculate the VIDVI for remote sensing imagery.
    Parameters:
        red (np.ndarray): NumPy array representing the Visible Red band.
        green (np.ndarray): NumPy array representing the Visible Green band.
        blue (float): NumPy array representing the Visible Blue band.
    Returns:
        np.ndarray: Computed VIDVI array.
    """
    red_f = red.astype(np.float32)
    green_f = green.astype(np.float32)
    blue_f = blue.astype(np.float32)
    num = 2 * green_f - red_f - blue_f
    denom = 2 * green_f + red_f + blue_f
    vidvi = safe_div(num, denom)
    return vidvi

def calculate_cive(red, green, blue):
    """
    Calculate the CIVE for remote sensing imagery.
    Parameters:
        red (np.ndarray): NumPy array representing the Visible Red band.
        green (np.ndarray): NumPy array representing the Visible Green band.
        blue (float): NumPy array representing the Visible Blue band.
    Returns:
        np.ndarray: Computed CIVE array.
    """
    red_f = red.astype(np.float32)
    green_f = green.astype(np.float32)
    blue_f = blue.astype(np.float32)
    cive = (0.441 * red_f) - (0.81 * green_f) + (0.385 * blue_f) + 18.7874
    return cive

def prep_segmentation(input_path, output_folder, nodata_val=0):
    """
    Prepare raster bands by computing vegetation indices (NDVI and NDREI) for segmentation.
    Parameters:
        input_path (str): Path to the input multiband raster file.
        output_folder (str): Directory where the transformed raster files will be saved.
        nodata_val (float, optional): Nodata value to set in the profile. Defaults to 0.
    Returns:
        tuple: File paths for the generated NDREI and NDVI raster files (ndrei_file_path, ndvi_file_path).
    """
    # Memuat band
    with rio.open(input_path) as src:
        m_red = src.read(5)
        red_edge = src.read(6)
        nir = src.read(7)
        profile = src.profile

    # print("\nMenghitung transformasi indeks vegetasi...")
    profile.update(
        dtype="float32",
        count=1
    )
    # Mentransformasi citra
    transform_ndvi = calculate_ndvi(nir, m_red)
    transform_ndrei = calculate_ndvi(nir, red_edge)
    ndvi_file_path = save_raster(transform_ndvi, profile, output_folder, "NDVI.tif", nodata_val)
    ndrei_file_path = save_raster(transform_ndrei, profile, output_folder, "NDREI.tif", nodata_val)
    return ndrei_file_path, ndvi_file_path 

def calc_vegetation_indices(input_path, output_folder):
    """
    Perform vegetation index transformation 
    especially for water model dataset.
    Parameters:
        input_path (str): Path to extraction result file (.csv).
        output_folder (str): Directory where the output file will be saved.
    Returns:
        str: Path to generated CSV file.
    """
    os.makedirs(output_folder, exist_ok=True)
    base_name = get_filename(input_path)
    output_path = os.path.join(output_folder, f"{base_name}_vegetation_indices.csv")
    df = pd.read_csv(input_path)
    bands = ["RED", "GREEN", "BLUE", "M_GREEN", "M_RED", "RED_EDGE", "NIR"]
    data_trimmed = trim_norm_df(df[bands].copy())
    cols_identitas = ["id", "Nama", "X", "Y"]

    # Terapkan urutan
    data_trimmed["CIVE"]  = calculate_cive(data_trimmed["RED"], data_trimmed["GREEN"], data_trimmed["BLUE"])
    data_trimmed["EVI"]   = calculate_evi(data_trimmed["NIR"], data_trimmed["M_RED"], data_trimmed["BLUE"])
    data_trimmed["GNDVI"] = calculate_gndvi(data_trimmed["NIR"], data_trimmed["M_GREEN"])
    data_trimmed["NDRE"]  = calculate_ndrei(data_trimmed["NIR"], data_trimmed["RED_EDGE"])
    data_trimmed["NDVI"]  = calculate_ndvi(data_trimmed["NIR"], data_trimmed["M_RED"]) 
    data_trimmed["VIDVI"] = calculate_vidvi(data_trimmed["RED"], data_trimmed["GREEN"], data_trimmed["BLUE"])

    for col in cols_identitas:
        data_trimmed[col] = df[col]

    cols_indeks = ["CIVE", "EVI", "GNDVI", "NDRE", "NDVI", "VIDVI"]
    urutan_final = cols_identitas + bands + cols_indeks
    data_final = data_trimmed[urutan_final]
    data_final.to_csv(output_path, index=False)
    return output_path

def segmentation(input_path, ndvi_path, output_folder, check_cancel=None, on_progress=None, nodata_val=0):
    """
    Perform raster segmentation using multiband raster and NDVI files to generate a threshold mask file.
    Parameters:
        input_path (str): Path to the input multiband raster file.
        ndvi_path (str): Path to NDVI raster file.
        output_folder (str): Directory where the output file will be saved.
        check_cancel (callable, optional): Optional callback function to check for cancellation requests.
        on_progress (callable, optional): Optional callback function to track processing progress.
        nodata_val (float): nodata_val (float, optional): Nodata value to set in the profile. Defaults to 0.
    Returns:
        str: Path to generated GeoTIff threshold file.
    """
    os.makedirs(output_folder, exist_ok=True)
    base_name = get_filename(input_path)
    # Membuat peta segmentasi gulma dan padi
    model_gulma = str(AppPaths.assets("defaults/models/segmenter_model.joblib"))
    peta_segmentasi_gulma = separate_weeds(
        model_gulma, 
        input_path, 
        output_folder, 
        check_cancel, 
        on_progress
    )
    # print("Memuat file hasil transformasi...")
    with (
        rio.open(ndvi_path) as src_ndvi,
        rio.open(peta_segmentasi_gulma) as src_gulma
    ):
        ndvi = src_ndvi.read(1).astype(float)
        mask_padi = src_gulma.read(1).astype(float) == 1
        profile = src_ndvi.profile
    
    # Menghitung threshold indeks vegetasi
    # t_ndre = otsu_threshold(ndre, jumlah_bin=256, rentang_nilai=(-1, 1))
    t_ndvi = otsu_threshold(ndvi, jumlah_bin=256, rentang_nilai=(-1, 1))
    logger.info(f"Threshold set: {t_ndvi}")
    # t_savi = otsu_threshold(savi, jumlah_bin=256, rentang_nilai=(-1, 1))

    # Thresholding
    mask_ndvi = ndvi > t_ndvi
    mask_final = mask_ndvi & mask_padi
    # print(f"Menghitung ambang NDVI dengan batas {t_ndvi}...")
    hasil_threshold = mask_final.astype(float)

    # Menyimpan hasil threshold
    threshold_file_path = save_raster(
        hasil_threshold, 
        profile, 
        output_folder, 
        f"{base_name}_threshold_result.tif", 
        nodata_val
    )
    os.remove(peta_segmentasi_gulma)
    return threshold_file_path