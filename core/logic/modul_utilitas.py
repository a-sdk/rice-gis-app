"""
Modul berisi fungsi-fungsi pembantu
yang digunakan pada program utama
"""

import os
import json
import numpy as np
import rasterio as rio
import geopandas as gpd
import fiona
import logging

logger = logging.getLogger(__name__)

def getRootFileName(input_path):
    base = os.path.splitext(os.path.basename(input_path))[0]
    return base.split('_')[0]

# Fungsi bantu membuat raster konstan
def create_constant_raster(tif, value, output_folder, output_filename):
    """
    Membuat array baru dengan ukuran sama seperti raster sumber.
    Semua piksel non-NoData diisi dengan `value`.
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)

    with rio.open(tif) as src:
        mask = src.read_masks(1) > 0
        profile = src.profile.copy()

    output = np.zeros((src.height, src.width), dtype=np.uint8)

    # Isi piksel valid
    output[mask] = value

    profile.update(
        count=1, 
        dtype=rio.uint8, 
        nodata=0
        ) 

    # Tulis ke file stack baru
    with rio.open(output_path, 'w', **profile) as dest:
        dest.write(output, 1)

    logger.info(f"Band hst berhasil dibuat")
    return output_path

# Fungsi bantu pembuatan multipoligon 
def lonlat_to_utm_epsg(lon, lat):
    from pyproj import database
    from pyproj.aoi import AreaOfInterest

    utm_crs_list = database.query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=lon,
            south_lat_degree=lat,
            east_lon_degree=lon,
            north_lat_degree=lat,
        ),
    )
    # Mengambil kode EPSG dari hasil pertama yang ditemukan
    return int(utm_crs_list[0].code)

def generate_grid_points(polygon, target_count):
    from shapely.geometry import Point
    bounds = polygon.bounds # (minx, miny, maxx, maxy)
    area = polygon.area
    
    # Hitung jarak antar titik ideal (asumsi persegi)
    step = (area / target_count) ** 0.5
    
    x_coords = np.arange(bounds[0], bounds[2], step)
    y_coords = np.arange(bounds[1], bounds[3], step)
    
    points = []
    for x in x_coords:
        for y in y_coords:
            p = Point(x, y)
            if polygon.contains(p):
                points.append(p)
    return points
        
# Fungsi untuk membuat multipoligon (shahiban)
def buat_multipoligon(shp_path, output_folder, on_progress=None, subpoly_area=0.5):
    """
    Membuat multipoligon dari shapefile poligon.

    Parameters:
        shp_path (str): Lokasi shapefile yang menjadi acuan.
        output_folder (str): Nama folder tempat hasil klip disimpan.
        
    Returns:
        str: Output path.
    """
    from sklearn.cluster import MiniBatchKMeans
    from shapely.geometry import Point, MultiPoint
    import shapely.ops as ops
    # print("\nMembuat multipoligon...")
    
    output_folder = f"{output_folder}/multipoligon"
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.splitext(os.path.basename(shp_path))[0]
    # print(f"Memproses file: {filename}.shp")
    
    # ========================================
    # Tahap 1: Membaca & Konversi CRS
    # ========================================
    if on_progress:
        on_progress(11, f"Loading polygon crs...")
    polygons = gpd.read_file(shp_path)
    crs_asal = polygons.crs.to_string()
    if polygons.crs is None:
        raise ValueError(f"File {filename} tidak memiliki CRS. Harap periksa file.")

    if polygons.crs.is_geographic:
        centroid = polygons.unary_union.centroid
        epsg_code = lonlat_to_utm_epsg(centroid.x, centroid.y)
        polygons = polygons.to_crs(epsg=epsg_code)
        logger.info(f"CRS diubah dari {crs_asal} ke UTM (EPSG:{epsg_code})")
    else:
        logger.info(f"CRS sudah proyeksi ({str(polygons.crs).upper()})")

    # Menentukan jumlah komponen poligon berdasarkan luas 
    with fiona.open(shp_path) as shp:
        jml_poligon = len(shp)
    polygons["area"] = polygons.area
    # Minimal satu komponen setiap poligon
    polygons["n_comp"] = np.maximum(
        1,
        np.ceil(polygons["area"] / subpoly_area).astype(int)
    )
    polygons["n_point"] = np.clip(
        polygons["n_comp"] * 50,
        400,
        8000
    ).astype(int)
    jml_cluster = polygons["n_comp"].sum()
    poly_area = polygons["area"].sum()
    logger.info(f"Terdapat {jml_poligon} poligon")
    logger.info(f"Total luas {poly_area:.2f} m2")
    logger.info(f"Jumlah cluster {jml_cluster}")
    logger.info(
    "\n%s",
    polygons[["id", "area", "n_comp", "n_point"]]
    )
    output_intersection = os.path.join(output_folder, f"{filename}_{jml_cluster}_part.shp")
    # ========================================
    # Tahap 2: Generate Points
    # ========================================
    if on_progress:
        on_progress(12, f"Generating points...")
    all_points = []
    for _, row in polygons.iterrows():
        pts = generate_grid_points(
            row.geometry,
            row.n_point
        )
        all_points.extend(pts)
    gdf_points = gpd.GeoDataFrame(geometry=all_points, crs=polygons.crs)
    logger.info(f"{len(gdf_points)} titik acak dihasilkan")

    # ========================================
    # Tahap 3: K-Means Clustering
    # ========================================
    if on_progress:
        on_progress(13, f"Clustering points...")

    coords = np.column_stack((gdf_points.geometry.x, gdf_points.geometry.y))
    if len(gdf_points) < jml_cluster:
        raise ValueError("Points count must be greater than cluster count.")

    batch_size = min(len(coords), max(1024, jml_cluster * 2))
    kmeans = MiniBatchKMeans(
        n_clusters=jml_cluster,
        random_state=42,
        batch_size=batch_size,
        n_init=3,
        max_iter=100,
    )
    gdf_points["CLUSTER_ID"] = kmeans.fit_predict(coords)
    centers = kmeans.cluster_centers_
    logger.info("K-Means clustering selesai")

    # ========================================
    # Tahap 4: Hitung Centroid
    # ========================================
    if on_progress:
        on_progress(15, f"Calculating centroids...")
    centroids = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in centers],
        crs=polygons.crs
    )
    logger.info(f"{len(centroids)} centroid dihasilkan")

    # ========================================
    # Tahap 5: Voronoi Polygon
    # ========================================
    if on_progress:
        on_progress(16, f"Generating voronoi polygons...")
    points = MultiPoint(list(centroids.geometry))
    # buffer_union = centroids.buffer(100).unary_union
    boundary = polygons.unary_union.envelope.buffer(500) # buffer_union.convex_hull

    vor = ops.voronoi_diagram(points, envelope=boundary, tolerance=0)
    polys = [poly.buffer(0) for poly in vor.geoms if poly.is_valid and not poly.is_empty]

    gdf_voronoi = gpd.GeoDataFrame(geometry=polys, crs=centroids.crs)
    gdf_voronoi = gpd.sjoin_nearest(
        gdf_voronoi, centroids, how="left", distance_col="dist"
    )

    # logger.info("Voronoi polygons selesai")

    # ========================================
    # Tahap 6: Intersection
    # ========================================
    if on_progress:
        on_progress(19, f"Performing intersection...")
    if polygons.crs != gdf_voronoi.crs:
        gdf_voronoi = gdf_voronoi.to_crs(polygons.crs)
        # logger.info("CRS berbeda, disamakan dulu.")

    gdf_inter = gpd.overlay(polygons, gdf_voronoi, how="intersection")
    gdf_inter = gdf_inter[
        ~gdf_inter.geometry.is_empty & gdf_inter.geometry.is_valid
    ]
    gdf_inter = gdf_inter.drop(columns=['index_right'])
    # Menyiapkan kolom untuk prediksi model
    gdf_inter["no_urut"] = gdf_inter.groupby("id").cumcount() + 1
    # === Simpan langsung ke folder utama tanpa subfolder ===
    gdf_inter.to_file(output_intersection, driver="ESRI Shapefile")

    logger.info(f"Intersection selesai")

    return output_intersection

# Fungsi untuk menyimpan array NumPy ke dalam file GeoTIFF
def simpan_raster(input_raster, profile, output_folder, output_filename, nilai_nodata=None):
    """
    Menyimpan array NumPy sebagai file GeoTIFF ke folder yang ditentukan.

    Parameters:
        input_raster (np.ndarray): Array NumPy yang akan disimpan.
        profile (dict): Metadata raster dari file sumber.
        output_folder (str): Nama folder tempat file akan disimpan.
        output_filename (str): Nama file output, termasuk ekstensi.
        
    Returns:
        str: Output path.
    """
    # Gabungkan nama folder dan nama file
    output_path = os.path.join(output_folder, output_filename)

    # Pastikan folder output ada, jika tidak, buat folder baru
    os.makedirs(output_folder, exist_ok=True)

    # Perbarui profil untuk file output
    profile.update(  
        count=1,
        driver="GTiff"
    )
    if nilai_nodata is not None:
        profile.update(nodata=nilai_nodata)
        
    with rio.open(output_path, 'w', **profile) as dest:
        dest.write(input_raster, 1)

    # print(f"File {output_filename} berhasil disimpan di {output_folder}")
    return output_path

# Fungsi untuk menentukan threshold secara otomatis
def otsu_threshold(input_raster, jumlah_bin=256, rentang_nilai=(-1, 1)):
    """
    Menghitung threshold untuk masking raster.

    Parameters:
        input_raster (np.ndarray): Data raster yang akan dihitung thresholdnya.
        jumlah_bin (int): Jumlah bin histogram.
        rentang_nilai (str): Nilai min dan max yang ada pada data raster.
        
    Returns:
        float: Nilai threshold.
    """
    # Histogram dan probabilitas
    hist, bin_edges = np.histogram(input_raster.ravel(), bins=jumlah_bin, range=rentang_nilai)
    prob = hist.astype(float) / hist.sum()

    # Bin center
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Probabilitas kumulatif (q1) dan mean kumulatif (mu1)
    q1 = np.cumsum(prob)
    mu1 = np.cumsum(prob * bin_mids)

    # Total mean
    global_mean = mu1[-1]

    # Kelas 2
    q2 = 1 - q1
    mu2 = global_mean - mu1

    # Hindari pembagian dengan nol
    valid = (q1 > 0) & (q2 > 0)

    mean1 = np.zeros_like(mu1)
    mean2 = np.zeros_like(mu1)

    mean1[valid] = mu1[valid] / q1[valid]
    mean2[valid] = mu2[valid] / q2[valid]

    # Variansi antar kelas
    selisih_var = q1 * q2 * (mean1 - mean2) ** 2

    # Ambil threshold dengan variansi antar kelas maksimum
    nilai_terbaik = np.argmax(selisih_var * valid) 
    threshold = bin_mids[nilai_terbaik]

    return threshold

# Fungsi menumpuk semua fitur (band)
def tumpuk_fitur(lst_fitur, output_folder, output_filename, nilai_nodata=0):
    """
    Menumpuk seluruh band atau fitur menjadi array Numpy besar.

    Parameters:
        lst_fitur (list): List fitur-fitur yang akan ditumpuk.
        output_folder (str): Nama folder tempat file akan disimpan.
        output_filename (str): Nama file output, termasuk ekstensi.
        nilai_nodata (float): Nilai nodata raster.

    Returns:
        str: Output path.
    """
    # Gabung nama folder dan file
    output_path = os.path.join(output_folder, output_filename)
    os.makedirs(output_folder, exist_ok=True)
    # Baca semua file dan kumpulkan datanya
    feature_stack = []
    profile = None 

    for fitur in lst_fitur:
        with rio.open(fitur) as src:
            if profile is None:
                profile = src.profile
                profile.update(
                    dtype="float32",
                    nodata=nilai_nodata
                ) 
            # Baca semua band dari file ini
            feature_stack.append(src.read())

    # Gabungkan semua data menjadi satu array NumPy besar
    full_stack_array = np.vstack(feature_stack)

    # Perbarui profile untuk file stack baru
    total_bands = full_stack_array.shape[0]
    profile.update(count=total_bands, nodata=nilai_nodata) 

    # Tulis ke file stack baru
    with rio.open(output_path, 'w', **profile) as dst:
        dst.write(full_stack_array)

    logger.info(f"Tumpukan fitur berhasil disimpan")
    return output_path

# Fungsi untuk menghitung sebaran penyakit per rumpun
def hitung_sebaran_rumpun(input_path, legend_dict):
    """
    Menghitung sebaran penyakit.
    
    Parameters:
        input_path (list): Lokasi file GeoTIFF hasil prediksi model.
        legend (dict): Informasi legenda.

    Returns:
        None.
    """
    
    from path_config import InfoRegistry
    for file in input_path:
        legend = {k: v.copy() for k, v in legend_dict.items()}
        nf = os.path.splitext(os.path.basename(file))[0]
        nf_penyakit = nf.split("_")[0]
        DISEASE_MAP = {
            "blas": "Leaf blast disease",
            "blb":  "Bacterial leaf blight disease",
            "bs":   "Brown spot disease",
            "nbs":  "Narrow brown spot disease"
        }
        penyakit = DISEASE_MAP.get(nf_penyakit, "Rice disease")
        with rio.open(file) as src:
            data = src.read(1)
            nodata = src.nodata
        # Mengecualikan nodata
        mask_valid = (data != nodata)
        valid_data = data[mask_valid]
        # Menghitung total piksel valid
        jml_data_valid = valid_data.size
        # Mengambil semua nilai unik
        counts = dict(zip(*np.unique(data, return_counts=True)))
        for key, item in legend.items():
            pct = round((counts.get(key, 0) / jml_data_valid) * 100, 1)
            item["pct"] = pct  
            item["label"] = f"{item['label']} ({pct}%)"

        # Mengambil nilai untuk logika
        p_sehat = legend[1]["pct"]
        p_ringan = legend[2]["pct"]
        p_sedang = legend[3]["pct"]
        p_parah = legend[4]["pct"]
        val_kacau = p_sedang + p_parah 
        # Logika Rekomendasi
        # print("-" * 30)
        if val_kacau > p_sehat and val_kacau > p_ringan:
            status_desc = InfoRegistry.get_info("disease", "severe")
            recom = InfoRegistry.get_recom("disease", "severe")
        elif p_ringan > p_sehat and p_ringan > val_kacau :  
            status_desc = InfoRegistry.get_info("disease", "low")
            recom = InfoRegistry.get_recom("disease", "low")
        else:
            status_desc = InfoRegistry.get_info("disease", "healthy")
            recom = InfoRegistry.get_recom("disease", "healthy")
        msg = f"{penyakit} is {status_desc}"
        info_dict = {
            "info": msg,
            "recom": recom
        }

        legend_json = json.dumps(legend)
        stats_json = json.dumps(info_dict)
        # Menyimpan legenda dan stats sebagai metadata
        with rio.open(file, "r+") as dest:
            dest.update_tags(
                LEGEND=legend_json,
                STATS=stats_json
                )
       
# Fungsi untuk menghitung sebaran per petak
def hitung_sebaran_petak(gpkg_path, legend_dict):
    """
    Menghitung sebaran penyakit.
    
    Parameters:
        gpkg_path (str): Lokasi shapefile hasil prediksi model.
        legend (dict): Informasi legenda.

    Returns:
        None.
    """
    import sqlite3
    from path_config import InfoRegistry
    try:
        layers = fiona.listlayers(gpkg_path)
    except Exception as e:
        logger.info(f"ERROR: {e}")
    # Siapkan tabel database gpkg
    conn = sqlite3.connect(gpkg_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS app_layer_metadata (
        layer_name TEXT PRIMARY KEY,
        legend_json TEXT,
        stats_json TEXT
    )
    """)
    
    # Hitung sebaran per nama layer
    for layer_name in layers:
        if layer_name == "app_layer_metadata":
            continue
        
        legend = {k: v.copy() for k, v in legend_dict.items()}
        # print(f"\n=== Menganalisis Layer: {layer_name} ===")
        # Membaca layer spesifik
        gdf = gpd.read_file(gpkg_path, layer=layer_name)
        
        if "preds" not in gdf.columns:
            logger.info(f"Kolom 'preds' tidak ditemukan di layer {layer_name}")
            continue

        data = gdf["preds"]
        jml_data = len(data)
        
        if jml_data == 0:
            logger.info(f"Tidak ada data pada layer {layer_name}")
            continue

        if layer_name == "water":
            # LOGIKA REGRESI
            for key, item in legend.items():
                low, high = item["range"]
                # Menghitung berapa banyak nilai yang masuk dalam rentang ini
                count = data[(data >= low) & (data < high)].count()
                pct = round((count / jml_data) * 100, 1)
                item["pct"] = pct  
                item["label"] = f"{item['label']} ({pct}%)"

        else:
            # LOGIKA KLASIFIKASI
            counts = data.value_counts()
            for key, item in legend.items():
                pct = round((counts.get(key, 0) / jml_data) * 100, 1)
                item["pct"] = pct  
                item["label"] = f"{item['label']} ({pct}%)"

        # Logika Rekomendasi
        val_1 = legend[1]["pct"]
        val_2 = legend[2]["pct"]
        val_3 = legend[3]["pct"]
        if 4 in legend:
            val_4 = legend[4]["pct"]
            val_5 = legend[5]["pct"]

        if len(legend) == 3: # Nutrisi (kurang, cukup, berlebih)
            if layer_name == "nitrogen":
                status_desc = InfoRegistry.get_info("nutrient", "nitrogen")
            elif layer_name == "phospor":
                status_desc = InfoRegistry.get_info("nutrient", "phospor")
            elif layer_name == "kalium":
                status_desc = InfoRegistry.get_info("nutrient", "kalium")
            msg = status_desc
            val_ = val_2 + val_3 # gabungan cukup + berlebih
            if val_1 > val_2 and val_1 > val_3:
                recom = InfoRegistry.get_recom("nutrient", "deficit")
            elif val_2 > val_1 and val_2 > val_3:
                recom = InfoRegistry.get_recom("nutrient", "adequate")
            else:
                recom = InfoRegistry.get_recom("nutrient", "excess")
        
        elif len(legend) == 4: # Penyakit (sehat, ringan, sedang, parah)
            DISEASE_MAP = {
                "blas": "Leaf blast disease",
                "blb":  "Bacterial leaf blight disease",
                "bs":   "Brown spot disease",
                "nbs":  "Narrow brown spot disease"
            }
            penyakit = DISEASE_MAP.get(layer_name, "Rice disease")
            val_ = val_3 + val_4 # gabungan sedang + parah

            if val_ > val_1 and val_ > val_2:
                status_desc = InfoRegistry.get_info("disease", "severe")
                recom = InfoRegistry.get_recom("disease", "severe")
                
            elif val_2 > val_1 and val_2 > val_: 
                status_desc = InfoRegistry.get_info("disease", "low")
                recom =  InfoRegistry.get_recom("disease", "low")
                 
            else:
                status_desc = InfoRegistry.get_info("disease", "healty")
                recom = InfoRegistry.get_recom("disease","healthy")
            msg = f"{penyakit.title()} is {status_desc}"
                
 
        elif len(legend) == 6: # Air (rentang)
            val_ = val_4 + val_5 
            if val_1 > val_2 + val_3 and val_1 > val_:
                recom = InfoRegistry.get_recom("water", "adequate")
            elif val_2 + val_3 > val_ and val_2 + val_3 > val_1:
                recom = InfoRegistry.get_recom("water", "mid")
            elif val_ > val_2 + val_3 and val_ > val_1:
                recom = InfoRegistry.get_recom("water", "dry")
            status_desc = InfoRegistry.get_info("water", "manage")
            msg = status_desc
        # Simpan hasil per layer ke dictionary utama
        info_dict = {
            "info": msg,
            "recom": recom
        }
        
        # Masukkan ke database
        cur.execute("""
        INSERT OR REPLACE INTO app_layer_metadata
        (layer_name, legend_json, stats_json)
        VALUES (?, ?, ?)
        """, (
            layer_name,
            json.dumps(legend),
            json.dumps(info_dict)
        ))

    # Menutup database
    conn.commit()
    conn.close()
    

if __name__ == "__main__":
    buat_multipoligon(
        shp_path=r"C:\Users\acer_\Documents\Shapefiles\cileunyi\petak rel.shp",
        output_folder=r"C:\Users\acer_\Documents\Shapefiles\cileunyi"
    )