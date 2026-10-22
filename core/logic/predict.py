import rasterio as rio
import os
from core.logic.utils import get_filename
import numpy as np
import onnxruntime as ort
import logging
import geopandas as gpd
import pandas as pd


logger = logging.getLogger(__name__)

def separate_weeds(model_path, input_path, output_folder, check_cancel, on_progress):
    """
    Apply a trained model to a raster to separate rice crops and weeds.
    Parameters:
        model_path (str): Path to the trained weed detection model file.
        input_path (str): Path to the input feature stack file.
        output_folder (str): Directory where the output file will be saved.
        check_cancel (callable, optional): Optional callback function to check for cancellation requests.
        on_progress (callable, optional): Optional callback function to track processing progress.
    Returns:
        str: Path to the generated output file.
    """
    import joblib
    base_name = get_filename(input_path)
    output_path = os.path.join(output_folder, f"{base_name}_segmented.tif")
    os.makedirs(output_folder, exist_ok=True)
    # print(f"Memuat model...")
    try:
        with joblib.parallel_backend('threading'):
            model = joblib.load(model_path)
    except FileNotFoundError:
        logger.error("Model not found")
        # print(f"ERROR: File model tidak ditemukan. Silahkan dicek dulu.")
        return

    # print(f"Membuka tumpukan fitur...")
    with rio.open(input_path) as src:
        # Dapatkan metadata untuk file output
        profile = src.profile
        nodata = src.nodata
        profile.update(
            dtype=rio.uint8,  
            count=1,               
            nodata=0           
        )
        
        # print(f"Memisahkan padi dengan gulma...")
        with rio.open(output_path, "w", **profile) as dest:
            # Proses citra dalam "potongan" (chunks/tiles) untuk menghemat RAM
            total_blocks = len(list(src.block_windows(1)))
            current_block = 0
            for ji, window in src.block_windows(1):
                current_block += 1
                # Memeriksa interupsi
                if check_cancel and check_cancel():
                    from app.worker import OperationCancelledError
                    logger.warning("Segmenter stopped")
                    raise OperationCancelledError("User cancelled")
                
                # Perbarui progres internal 
                if on_progress and current_block % 10 == 0:
                    relative_prog = 40 + int((current_block/total_blocks) * 10)
                    on_progress(relative_prog, f"Separating vegetation ({current_block}/{total_blocks})")
                # Baca data untuk potongan ini
                stack_chunk = src.read(window=window)
                
                # Reshape 3D (bands, rows, cols) -> 2D (pixels, features)
                # Pindahkan sumbu bands ke akhir: (rows, cols, bands)
                img_reshaped = np.moveaxis(stack_chunk, 0, -1)
                # Ratakan: (rows*cols, bands)
                pixels_flat = img_reshaped.reshape(-1, src.count)
                
                # Cari piksel yang valid (bukan NoData)
                valid_mask = pixels_flat[:, 6] != nodata
                pixels_valid = pixels_flat[valid_mask]
                
                # Siapkan kanvas hasil untuk potongan ini
                result_chunk = np.zeros(pixels_flat.shape[0], dtype=rio.uint8)
                
                # Lakukan Prediksi (Hanya pada piksel valid)
                if pixels_valid.shape[0] > 0:
                    predictions = model.predict(pixels_valid)
                    
                    # Isi kanvas hasil
                    result_chunk[valid_mask] = predictions
                
                # Bentuk kembali 1D -> 2D dan tulis ke file output
                result_chunk_2d = result_chunk.reshape(window.height, window.width)
                dest.write(result_chunk_2d.astype(rio.uint8), window=window, indexes=1)
                import json
                isThreshold = json.dumps(True)
                dest.update_tags(THRESHOLD=isThreshold)
                
    # print(f"Pemisahan selesai! Peta segmentasi disimpan di: {output_folder}")
    return output_path

def predict_disease(scaler, model, input_path, output_folder, check_cancel, on_progress):
    """
    Apply a multi-output machine learning model to a raster feature stack to generate 4 separate disease distribution maps.
    Parameters:
        scaler (object): Fitted scaler object used to normalize input features.
        model (object): Trained multi-output prediction model.
        input_path (str): Path to the input raster file or feature stack.
        output_folder (str): Directory where the resulting distribution maps will be saved.
        check_cancel (callable, optional): Optional callback function to check for cancellation requests.
        on_progress (callable, optional): Optional callback function to track processing progress.
    Returns:
        list or str: Output path(s) to the generated prediction raster files.
    """
    session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_output_names = [output.name for output in session.get_outputs()]
    output_names = ["blas", "blb", "bs", "nbs"] 
    path_hasil = []
    import json
    isPrediction = json.dumps(True)
    # print(f"Memprediksi citra...")
    with rio.open(input_path) as src:
        base_profile = src.profile
        base_profile.update(
            dtype=rio.uint8, 
            count=1,          
            nodata=0
        )
        output_dests = {}
        output_folder = f"{output_folder}/Preds_Result"
        os.makedirs(output_folder, exist_ok=True)
        for i, name in enumerate(output_names):
            output_path = os.path.join(output_folder, f"{name}_prediction_map.tif")
            path_hasil.append(output_path)
            output_profile = base_profile.copy()
            if os.path.exists(output_path):
                output_dests[name] = rio.open(output_path, "r+")
            else:
                output_dests[name] = rio.open(output_path, "w", **output_profile)
        total_blocks = len(list(src.block_windows(1)))
        current_block = 0
        # Memproses lewat potongan (chunk)
        for ji, window in src.block_windows(1):
            current_block += 1
            # Memeriksa interupsi
            if check_cancel and check_cancel():
                from app.worker import OperationCancelledError
                logger.warning("Classifier stopped")
                raise OperationCancelledError("User cancelled")
            
            # Perbarui progres internal 
            if on_progress and current_block % 10 == 0:
                relative_prog = 70 + int((current_block/total_blocks) * 20)
                on_progress(relative_prog, f"Generating prediction ({current_block}/{total_blocks})")
            
            # print(f"Memproses potongan di {window}...")
            
            # Baca per potongan
            stack_chunk = src.read(window=window)
            
            # Reshape (bands, rows, cols) -> (pixels, features)
            img_reshaped = np.moveaxis(stack_chunk, 0, -1)
            pixels_flat = img_reshaped.reshape(-1, src.count)
            
            # Menentukan piksel yang valid
            valid_mask = (pixels_flat[:, 0] != src.nodata)
            pixels_valid = pixels_flat[valid_mask]
            
            # Memprediksi piksel 
            if pixels_valid.shape[0] > 0:
                pixels_valid_scaled = scaler.transform(pixels_valid)
                pixels_valid_scaled = np.asarray(pixels_valid_scaled, dtype=np.float32)
                predictions_list = session.run(
                    onnx_output_names,
                    {input_name: pixels_valid_scaled.astype(np.float32)}
                )
            else:
                predictions_list = [np.array([])] * len(output_names) 

            # Loop setiap output
            for i, name in enumerate(output_names):
                
                result_chunk = np.zeros(pixels_flat.shape[0], dtype=rio.uint8)
                
                if predictions_list[i].size > 0:
                    labels_kelas = predictions_list[i].argmax(axis=1) 
                    
                    result_chunk[valid_mask] = labels_kelas + 1 
                
                # Reshape 1D -> 2D
                result_chunk_2d = result_chunk.reshape(window.height, window.width)
                
                # Menyimpan hasil prediksi
                dest = output_dests[name]
                dest.write(result_chunk_2d, window=window, indexes=1)
                dest.update_tags(PREDICTION=isPrediction)
                
        # Menutup semua file output setelah loop selesai
        for dest in output_dests.values():
            dest.close()
                
    # print(f"\nDeteksi selesai! 4 peta disimpan di folder: {output_folder}")
    return path_hasil

def predict_disease_plot(scaler, model, input_path, shp_path, output_folder, check_cancel, on_progress):
    """
    Apply a multi-output machine learning model to generate 4 separate disease distribution plots.
    Parameters:
        scaler (object): Fitted scaler object used to normalize input features.
        model (object): Trained multi-output prediction model.
        input_path (str): Path to the input CSV file.
        shp_path (str): Path to the reference shapefile containing polygon geometries.
        output_folder (str): Directory where the resulting distribution plots will be saved.
        check_cancel (callable, optional): Optional callback function to check for cancellation requests.
        on_progress (callable, optional): Optional callback function to track processing progress.
    Returns:
        str: Path to the generated prediction GeoPackage files.
    """
    df = pd.read_csv(input_path)
    gdf = gpd.read_file(shp_path)
    session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_output_names = [output.name for output in session.get_outputs()]
    fitur = ["RED", "GREEN", "BLUE", "M_GREEN", "M_RED", "RED_EDGE", "NIR", "NDREI"] 

    X_inf = df[fitur]
    X_inf_array = X_inf.values
    X_inf_scaled = scaler.transform(X_inf_array)
    X_inf_scaled = np.asarray(X_inf_scaled, dtype=np.float32)
    raw_preds = session.run(
        onnx_output_names,
        {input_name: X_inf_scaled.astype(np.float32)}
        )[0]
    
    disease_names = ["blas", "blb", "bs", "nbs"]
    map_label = {1: "Sehat", 2: "Ringan", 3: "Sedang", 4: "Parah"}

    output_folder = f"{output_folder}/Preds_Result/Plot/Disease"
    os.makedirs(output_folder, exist_ok=True)
    output_xlsx = os.path.join(output_folder, "Preds_Result.xlsx")
    output_shp = os.path.join(output_folder, "Preds_Result.shp")
    output_gpkg = os.path.join(output_folder, "Preds_Result.gpkg")
    gdf_single = gdf.copy()
    count = 0
    for i, name in enumerate(disease_names):
        count += 1
        # Memeriksa interupsi
        if check_cancel and check_cancel():
            from app.worker import OperationCancelledError
            logger.warning("Classifier dihentikan")
            raise OperationCancelledError("User cancelled")
            
        # Perbarui progres internal 
        if on_progress:
            relative_prog = 70 + int((count/len(disease_names)) * 20)
            on_progress(relative_prog, f"Generating prediction ({count}/{len(disease_names)})...")

        # Ambil index kelas tertinggi
        class_idx = np.argmax(raw_preds[i], axis=1) + 1
        # Masukkan ke dataframe
        df[f"Prediksi_{name}"] = [map_label[idx] for idx in class_idx]
        gdf[name] = [idx for idx in class_idx]
        gdf_single["preds"] = class_idx
        gdf_single.to_file(output_gpkg,
                            layer=name,
                            driver="GPKG"
                            )

    # print(f"Menyimpan hasil prediksi di {output_folder}")
    df.to_excel(output_xlsx, index=False)
    gdf.to_file(output_shp, driver="ESRI Shapefile")
    
    return output_gpkg

def predict_water_plot(scaler, model_reg, input_path, shp_path, output_folder, check_cancel, on_progress):
    """
    Apply a machine learning model to generate water adequacy distribution plots.
    Parameters:
        scaler (object): Fitted scaler object used to normalize input features.
        model_reg (object): Trained  prediction model.
        input_path (str): Path to the input CSV file.
        shp_path (str): Path to the reference shapefile containing polygon geometries.
        output_folder (str): Directory where the resulting distribution plots will be saved.
        check_cancel (callable, optional): Optional callback function to check for cancellation requests.
        on_progress (callable, optional): Optional callback function to track processing progress.
    Returns:
        str: Path to the generated prediction GeoPackage files.
    """
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    df = pd.read_csv(input_path)
    gdf = gpd.read_file(shp_path) 
    session = ort.InferenceSession(model_reg, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_output_names = [output.name for output in session.get_outputs()]
    fitur = ["M_GREEN", "M_RED", "NDVI", "NDRE", "GNDVI", "EVI", "VIDVI", "CIVE"] 
    X_inf = df[fitur]
    X_poly = poly.fit_transform(X_inf)
    X_poly_scaled = scaler.transform(X_poly)
    X_poly_scaled = np.asarray(X_poly_scaled, dtype=np.float32)
    reg_raw_preds = np.round(session.run(
        onnx_output_names, 
        {input_name: X_poly_scaled.astype(np.float32)}
    )[0].flatten(), 4)
    # model = ["klasifikasi", "regresi"]
    map_label = {1: "Cukup", 2: "Kurang"}
    output_folder = f"{output_folder}/Preds_Result/Plot/Water"
    # preds = [class_idx, reg_raw_preds]
    os.makedirs(output_folder, exist_ok=True)
    output_xlsx = os.path.join(output_folder, "Preds_Result.xlsx")
    output_shp = os.path.join(output_folder, "Preds_Result.shp")
    output_gpkg = os.path.join(output_folder, "Preds_Result.gpkg")
    gdf_single = gdf.copy()
    for i in range(5):
        # Memeriksa interupsi
        if check_cancel and check_cancel():
            from app.worker import OperationCancelledError
            logger.warning("Classifier stopped")
            raise OperationCancelledError("User cancelled")
            
        # Perbarui progres internal 
        if on_progress:
            relative_prog = 70 + int(((i+1)/5) * 20)
            on_progress(relative_prog, f"Generating prediction ({i+1}/5)...")
        # Masukkan ke dataframe
        df[f"prediksi_air_regresi "] = reg_raw_preds
        reg_preds = pd.Series(reg_raw_preds, index=df.index)
        gdf["regresi"] = reg_preds
        gdf_single["preds"] = reg_preds
        gdf_single.to_file(output_gpkg,
                            layer="water",
                            driver="GPKG"
                            )
        
    # print(f"Menyimpan hasil prediksi di {output_folder}")
    df.to_excel(output_xlsx, index=False)
    gdf.to_file(output_shp, driver="ESRI Shapefile")
    
    return output_gpkg
    
def predict_nutrient_plont(scaler_n, scaler_p, scaler_k, model_n, model_p, model_k, input_path, shp_path, output_folder, check_cancel, on_progress):
    """
    Apply a machine learning model to generate nutrient adequacy distribution plots.
    Parameters:
        scaler (object): Fitted scaler object used to normalize input features.
        model (object): Trained  prediction model.
        input_path (str): Path to the input CSV file.
        shp_path (str): Path to the reference shapefile containing polygon geometries.
        output_folder (str): Directory where the resulting distribution plots will be saved.
        check_cancel (callable, optional): Optional callback function to check for cancellation requests.
        on_progress (callable, optional): Optional callback function to track processing progress.
    Returns:
        str: Path to the generated prediction GeoPackage files.
    """
    df = pd.read_csv(input_path)
    gdf = gpd.read_file(shp_path)
    session_n = ort.InferenceSession(model_n, providers=["CPUExecutionProvider"])
    session_p = ort.InferenceSession(model_p, providers=["CPUExecutionProvider"])
    session_k = ort.InferenceSession(model_k, providers=["CPUExecutionProvider"])
    input_name_n = session_n.get_inputs()[0].name
    input_name_p = session_p.get_inputs()[0].name
    input_name_k = session_k.get_inputs()[0].name
    onnx_output_names_n = [output.name for output in session_n.get_outputs()]
    onnx_output_names_p = [output.name for output in session_p.get_outputs()]
    onnx_output_names_k = [output.name for output in session_k.get_outputs()]
    fitur = ["RED", "GREEN", "BLUE", "M_GREEN", "M_RED", "RED_EDGE", "NIR"] 

    X_inf = df[fitur]
    X_inf_array = X_inf.values
    X_n_scaled = scaler_n.transform(X_inf_array)
    X_p_scaled = scaler_p.transform(X_inf_array)
    X_k_scaled = scaler_k.transform(X_inf_array)
    X_n_scaled = np.asarray(X_n_scaled, dtype=np.float32)
    X_p_scaled = np.asarray(X_p_scaled, dtype=np.float32)
    X_k_scaled = np.asarray(X_k_scaled, dtype=np.float32)
    n_preds = session_n.run(
        onnx_output_names_n,
        {input_name_n: X_n_scaled.astype(np.float32)}
        )[0]
    p_preds = session_p.run(
        onnx_output_names_p,
        {input_name_p: X_p_scaled.astype(np.float32)}
    )[0]
    k_preds = session_k.run(
        onnx_output_names_k,
        {input_name_k: X_k_scaled.astype(np.float32)}
    )[0]
    # Ambil index kelas tertinggi
    n_class_idx = np.argmax(n_preds, axis=1) + 1
    p_class_idx = np.argmax(p_preds, axis=1) + 1
    k_class_idx = np.argmax(k_preds, axis=1) + 1
    
    map_label = {1: "Kurang", 2: "Cukup", 3: "Berlebih"}
    nutrient = ["nitrogen", "phospor", "kalium"]
    output_folder = f"{output_folder}/Preds_Result/Plot/Nutrient"
    os.makedirs(output_folder, exist_ok=True)
    output_xlsx = os.path.join(output_folder, "Preds_Result.xlsx")
    output_shp = os.path.join(output_folder, "Preds_Result.shp")
    output_gpkg = os.path.join(output_folder, "Preds_Result.gpkg")
    gdf_single = gdf.copy()
    prediction = [n_class_idx, p_class_idx, k_class_idx]
    for i, name in enumerate(nutrient):
        # Memeriksa interupsi
        if check_cancel and check_cancel():
            from app.worker import OperationCancelledError
            logger.warning("Classifier stopped")
            raise OperationCancelledError("User cancelled")
            
        # Perbarui progres internal 
        if on_progress:
            relative_prog = 70 + int(((i+1)/len(nutrient)) * 20)
            on_progress(relative_prog, f"Generating prediction ({i+1}/{len(nutrient)})...")

        # Masukkan ke dataframe
        df[f"Prediksi_{name}"] = [map_label[idx] for idx in prediction[i]]
        gdf[name] = pd.Series(prediction[i], index=df.index)
        gdf_single["preds"] = pd.Series(prediction[i], index=df.index)
        gdf_single.to_file(output_gpkg,
                            layer=name,
                            driver="GPKG"
                            )

    # print(f"Menyimpan hasil prediksi di {output_folder}")
    df.to_excel(output_xlsx, index=False)
    gdf.to_file(output_shp, driver="ESRI Shapefile")
    
    return output_gpkg

