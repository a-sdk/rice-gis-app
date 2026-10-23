"""
Utility functions module
"""

import os
import json
import numpy as np
import rasterio as rio
import geopandas as gpd
import fiona
import logging

logger = logging.getLogger(__name__)

def get_filename(path):
    """
    Get file name from a path.
    Parameters:
        path (str): Path to a file. 
    Returns:
        str: Filename.
    """  
    base = os.path.splitext(os.path.basename(path))[0]
    name = base.split('_')[0]
    return name

def lonlat_to_utm_epsg(lon, lat):
    """
    Convert geographic (lat/lon) coordinates to UTM projection.
    Parameters:
        lon (float): Longitude.
        lat (float): Latitude.  
    Returns:
        int: EPSG code.
    """

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
    
    return int(utm_crs_list[0].code)

def check_vector_crs(gdf):
    """
    Check vector CRS. If geographic, reprojects it to the local UTM zone.
    Parameters:
        gdf (GeoDataframe): Polygon or geometry.
    Returns:
        None.
    """
    crs_asal = gdf.crs.to_string()
    if gdf.crs is None:
        raise ValueError(f"No CRS found. Please check.")

    if gdf.crs.is_geographic:
        centroid = gdf.unary_union.centroid
        epsg_code = lonlat_to_utm_epsg(centroid.x, centroid.y)
        gdf = gdf.to_crs(epsg=epsg_code)
        logger.info(f"CRS converted: {crs_asal} -> UTM (EPSG:{epsg_code})")
    else:
        logger.info(f"CRS already projected ({str(gdf.crs).upper()})")

def check_and_reproject_raster(input_path, output_folder, check_cancel=None, on_progress=None):
    """
    Checks raster CRS. If geographic, reprojects it to the local UTM zone.
    Parameters:
        input_path (str): Path to a raster.
        output_folder (str): Directory where the reprojected raster will be saved.
        check_cancel (callable, optional): Optional callback function to check for cancellation requests.
        on_progress (callable, optional): Optional callback function to track processing progress.
    Returns:
        str: Path to reprojected raster.
    """
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    os.makedirs(output_folder, exist_ok=True)
    name = get_filename(input_path)
    output_path = os.path.join(output_folder, f"{name}_projected.tif")
    with rio.open(input_path) as src:
        if not src.crs or not src.crs.is_geographic:
            logger.info(f"CRS already projected or missing ({str(src.crs).upper()})")
            return input_path
        # Find the center point of the raster to determine UTM zone
        bounds = src.bounds
        center_lon = (bounds.left + bounds.right) / 2
        center_lat = (bounds.bottom + bounds.top) / 2
        
        # Get target UTM EPSG using your existing function
        epsg_code = lonlat_to_utm_epsg(center_lon, center_lat)
        target_crs = f"EPSG:{epsg_code}"
        
        logger.info(f"CRS converting: {src.crs} -> UTM ({target_crs})")
        
        # Calculate the new dimensions and transformation matrix
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *bounds
        )
        
        # Copy metadata and update it for the new projected raster
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
            
        # Reproject and write pixel data to the output file
        with rio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):

                if check_cancel and check_cancel():
                    from app.worker import OperationCancelledError
                    logger.warning("Reprojection stopped")
                    raise OperationCancelledError("User cancelled")

                if on_progress:
                    relative_prog = 30 + int((i/src.count) * 15)
                    on_progress(relative_prog, f"Reprojecting raster ({i}/{src.count})")

                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest # Use bilinear or cubic for continuous data
                )

    return output_path
            
def generate_grid_points(geom, target_count):
    """
    Generate points in a regular grid pattern within a given geometry.
    Parameters:
        geom (shapely.geometry.base.BaseGeometry): Polygon or geometry within which to generate points.
        target_count (int): Approximate target number of points to generate.
    Returns:
        list: List of shapely Point objects within the geometry.
    """
    from shapely.geometry import Point
    bounds = geom.bounds # (minx, miny, maxx, maxy)
    area = geom.area
    
    step = (area / target_count) ** 0.5
    
    x_coords = np.arange(bounds[0], bounds[2], step)
    y_coords = np.arange(bounds[1], bounds[3], step)
    
    points = []
    for x in x_coords:
        for y in y_coords:
            p = Point(x, y)
            if geom.contains(p):
                points.append(p)
    return points
        
def generate_voronoi_poly(shp_path, output_folder, on_progress=None, subpoly_area=0.45):
    """
    Subdivide polygons in a shapefile into smaller Voronoi polygons using K-Means clustering.
    Parameters:
        shp_path (str): Path to the input polygon shapefile (.shp).
        output_folder (str): Directory where the output shapefile will be saved.
        on_progress (callable, optional): Optional callback function to track processing progress.
        subpoly_area (float, optional): Approximate target area per subpolygon component in square meters.
            Defaults to 0.45.
    Returns:
        str: Path to generated intersected Voronoi shapefile.
    """
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from shapely.geometry import Point, MultiPoint
    import shapely.ops as ops
    # print("\nMembuat multipoligon...")
    
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.splitext(os.path.basename(shp_path))[0]
    # print(f"Memproses file: {filename}.shp")
    
    # ========================================
    # Tahap 1: Membaca & Konversi CRS
    # ========================================
    if on_progress:
        on_progress(11, f"Loading polygon CRS...")
    polygons = gpd.read_file(shp_path)
    
    check_vector_crs(polygons)
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
    logger.info(f"Polygon count: {jml_poligon}")
    logger.info(f"Total area: {poly_area:.2f} m2")
    logger.info(f"Available cluster count: {jml_cluster}")
    logger.info(
    "\n%s",
    polygons[["id", "area", "n_comp", "n_point"]]
    )
    output_intersection = os.path.join(output_folder, f"{filename}_voronoi.shp")
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
    logger.info(f"{len(gdf_points)} random points generated")

    # ========================================
    # Tahap 3: K-Means Clustering
    # ========================================
    if on_progress:
        on_progress(13, f"Clustering points...")

    coords = np.column_stack((gdf_points.geometry.x, gdf_points.geometry.y))
    if len(gdf_points) < jml_cluster:
        raise ValueError("Points count must be greater than cluster count.")

    # batch_size = min(len(coords), max(1024, jml_cluster * 2))
    # kmeans = MiniBatchKMeans(
    #     n_clusters=jml_cluster,
    #     random_state=42,
    #     batch_size=batch_size,
    #     n_init=3,
    #     max_iter=100,
    # )
    kmeans = KMeans(n_clusters=jml_cluster, random_state=42, n_init=20)
    gdf_points["CLUSTER_ID"] = kmeans.fit_predict(coords)
    centers = kmeans.cluster_centers_
    logger.info("K-Means clustering done")

    # ========================================
    # Tahap 4: Hitung Centroid
    # ========================================
    if on_progress:
        on_progress(15, f"Calculating centroids...")
    centroids = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in centers],
        crs=polygons.crs
    )
    logger.info(f"Generated {len(centroids)} centroid")

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

    # ========================================
    # Tahap 6: Intersection
    # ========================================
    if on_progress:
        on_progress(19, f"Performing intersection...")
    if polygons.crs != gdf_voronoi.crs:
        gdf_voronoi = gdf_voronoi.to_crs(polygons.crs)

    gdf_inter = gpd.overlay(polygons, gdf_voronoi, how="intersection")
    gdf_inter = gdf_inter[
        ~gdf_inter.geometry.is_empty & gdf_inter.geometry.is_valid
    ]
    gdf_inter = gdf_inter.drop(columns=['index_right'])
    # Menyiapkan kolom untuk prediksi model
    gdf_inter["no_urut"] = gdf_inter.groupby("id").cumcount() + 1
    # === Simpan langsung ke folder utama tanpa subfolder ===
    gdf_inter.to_file(output_intersection, driver="ESRI Shapefile")

    logger.info(f"Intersection performed succesfully")

    return output_intersection

def save_raster(input_raster, profile, output_folder, output_filename, nodata_val=None):
    """
    Save a NumPy array as a GeoTIFF file to the specified directory.
    Parameters:
        input_raster (np.ndarray): NumPy array to be saved.
        profile (dict): Raster metadata from the source file.
        output_folder (str): Directory where the file will be saved.
        output_filename (str): Output filename, including extension.
        nodata_val (float, optional): Nodata value to set in the profile. Defaults to None.   
    Returns:
        str: Path to the generated raster file.
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
    if nodata_val is not None:
        profile.update(nodata=nodata_val)
        
    with rio.open(output_path, 'w', **profile) as dest:
        dest.write(input_raster, 1)

    # print(f"File {output_filename} berhasil disimpan di {output_folder}")
    return output_path

def otsu_threshold(input_raster, jumlah_bin=256, rentang_nilai=(-1, 1)):
    """
    Calculate an optimal global threshold for raster masking using Otsu's method.
    Parameters:
        input_raster (np.ndarray): Input raster data array to calculate the threshold for.
        jumlah_bin (int, optional): Number of histogram bins. Defaults to 256.
        rentang_nilai (tuple of float, optional): Minimum and maximum intensity value range (min, max).
            Defaults to (-1, 1).
    Returns:
        float: Computed threshold value separating the foreground and background.
    """
    
    hist, bin_edges = np.histogram(input_raster.ravel(), bins=jumlah_bin, range=rentang_nilai)
    prob = hist.astype(float) / hist.sum()

    # Bin center
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2

    q1 = np.cumsum(prob)
    mu1 = np.cumsum(prob * bin_mids)

    # Total mean
    global_mean = mu1[-1]

    q2 = 1 - q1
    mu2 = global_mean - mu1

    valid = (q1 > 0) & (q2 > 0)

    mean1 = np.zeros_like(mu1)
    mean2 = np.zeros_like(mu1)

    mean1[valid] = mu1[valid] / q1[valid]
    mean2[valid] = mu2[valid] / q2[valid]

    selisih_var = q1 * q2 * (mean1 - mean2) ** 2

    nilai_terbaik = np.argmax(selisih_var * valid) 
    threshold = bin_mids[nilai_terbaik]

    return threshold

def stack_bands(lst_fitur, output_folder, output_filename, nodata_val=0):
    """
    Stack separate bands into a single multibands float32 raster file.
    Parameters:
        lst_fitur (list): List of raster bands to be stacked.
        output_folder (str): Directory where the output file will be saved.
        output_filename (str): Output filename, including extension.
        nodata_val (float, optional): Nodata value for the raster. Defaults to 0.
    Returns:
        str: Path to the generated raster file.
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
                    nodata=nodata_val
                ) 
            # Baca semua band dari file ini
            feature_stack.append(src.read())

    # Gabungkan semua data menjadi satu array NumPy besar
    full_stack_array = np.vstack(feature_stack)

    # Perbarui profile untuk file stack baru
    total_bands = full_stack_array.shape[0]
    profile.update(count=total_bands, nodata=nodata_val) 

    # Tulis ke file stack baru
    with rio.open(output_path, 'w', **profile) as dst:
        dst.write(full_stack_array)

    logger.info(f"Stacked raster saved succesfully")
    return output_path

def calculate_cluster_distribution(input_path, legend_dict):
    """
    Calculate the distribution and percentage of classes from raster prediction files.
    Parameters:
        input_path (list): List of file paths to model prediction GeoTIFF files.
        legend_dict (dict): Dictionary containing legend information and mapping classes.
    Returns:
        None
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
       
def calculate_plot_distribution(gpkg_path, legend_dict):
    """
    Calculate the distribution and percentage of classes from GeoPackage prediction files.
    Parameters:
        gpkg_path (str): File paths to model prediction GeoPackage files.
        legend (dict): Dictionary containing legend information and mapping classes.
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
            logger.info(f"No 'preds' column in {layer_name}")
            continue

        data = gdf["preds"]
        jml_data = len(data)
        
        if jml_data == 0:
            logger.info(f"Layer {layer_name}is empty")
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
