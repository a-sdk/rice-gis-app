# Changelogs

## Alpha version changelogs
### Version 1.4.11 - January 31, 2026
* **Bug Fixes**:
    * Removed the rebuild process for `float32` formatted images.
    * Combined 7 spectral bands and 4 transformation outputs using a more efficient workflow.

### Version 1.4.10 - January 22, 2026
* **New Features**:
    * Added a function to extract vertex coordinates in `ekstraksi.py`.
    * Added X (longitude) and Y (latitude) coordinate columns to the pixel extraction function.
    * Added a function to create multipolygons.
    * Added a function to extract vertex coordinates.
    * Added a function to generate disease distribution maps.

### Version 1.4.9 - December 26, 2025
* **New Features**:
    * Added functions for processing labels and datasets in `utils.py`.
* **Bug Fixes**:
    * Fixed nodata value errors in `klasifikasi.py`.
    * Fixed incorrect component ordering of extracted results in the `ekstrak_rerata_piksel()` function.

### Version 1.4.8 - December 13, 2025
* **New Features**:
    * Created a function to detect rice crop diseases.
    * Created the first version of the disease detection module in `deteksi.py`.
* **Bug Fixes**:
    * Fixed a memory leak issue in the `clip_raster()` function.

### Version 1.4.7 - November 20, 2025
* **New Features**:
    * Created a function to extract average pixel values from multipolygons.

### Version 1.4.6 - November 08, 2025
* **New Features**:
    * Added support for batch processing.
    * Displayed elapsed time per individual process as well as overall execution time.
    * Modified the `clip_raster()` function to support BigTIFF operations.

### Version 1.4.5 - November 03, 2025
* **New Features**:
    * Created two workflow options: separated features or stacked features.
    * Encapsulated the pixel value extraction process into dedicated functions.
    * Reorganized output directories for each processing step.
    * Moved utility functions to `utils.py`.
    * Separated each band into individual files along with index transformation outputs.
    * Modified `mask_raster()` to `mask_band_terpisah()` to enable masking across separated band files.
    * Updated workflow sequence to: 
        clip -> transform -> segment -> mask -> extract.
    * Updated output folder paths according to the new workflow.
    * Updated masking and extraction functions to display total valid pixels and extracted pixel counts.
    * Added progress bar visualization to the `ekstrak_tumpukan_fitur()` function.
    * Encapsulated transformation and segmentation processes into functions within `transformasi.py`.
* **Bug Fixes**:
    * Modified `clip_raster()` to prevent pixel values equal to 0 from being lost.

### Version 1.4.4 - October 30, 2025
* **New Features**:
    * Modified `ambil_file()` to display relevant file lists.
    * Added `tumpuk_fitur()` function to stack all features from a list.
    * Added `model_random_forest_0.joblib` for rice and weed classification.
    * Added `segmentasi_gulma.py` to run rice and weed classification.
* **Bug Fixes**:
    * Changed nodata values to NaN from -9999 to avoid processing errors.

### Version 1.4.3 - October 22, 2025
* **New Features**:
    * Made project structure flexible based on user input.
    * Encapsulated the histogram visualization into `tampilkan_histogram()`.
    * Added a display section for clipping results.
* **Bug Fixes**:
    * Removed black bounding boxes surrounding clipped outputs.
    * Fixed a bug causing spikes in value 0 on histograms due to nodata values computed during transformations.

### Version 1.4.2 - October 21, 2025
* **New Features**:
    * Encapsulated folder/file prompt procedures into `ambil_file()`.
    * Added NDREI as a thresholding reference to separate crops from weeds.
* **Bug Fixes**:
    * Fixed an issue where execution continued despite invalid folder or file paths.

### Version 1.4.1 - October 13, 2025
* **New Features**:
    * Added pixel value extraction based on polygon vertices.
    * Expanded extraction functionality to process all bands within a directory.
    * Saved extraction results directly to a `.csv` file.
    * Added progress bars to track pixel extraction progress.
* **Bug Fixes**:
    * Updated output directory locations for masking and transformation results to streamline extraction.
    * Resolved file overwrite issues.
    * Fixed NoData values being exported to `.csv`.
    * Reordered extraction column headers to meet project specifications.
    * Fixed inconsistent image data formatting.

### Version 1.4 - October 11, 2025
* **New Features**:
    * Added capability for users to define a custom working directory outside the project root.
    * Updated output file naming conventions to match processed input names dynamically.
    * Created `program_arogansi.py` for batch file processing experiments.

### Version 1.3 - October 08, 2025
* **New Features**:
    * Added user input selection for choosing input files.
    * Updated raster saving functionality to prevent overwriting existing files.
    * Added auto-thresholding features based on SAVI images.
* **Bug Fixes**:
    * Fixed bugs in the SAVI transformation function.

### Version 1.2 - September 09, 2025
* **New Features**:
    * Added raster clipping function from polygon shapefiles.
    * Added multi-spectral band masking across all channels.
    * Added utility function to inspect raster dimensions.

### Version 1.1 - August 07, 2025
* **New Features**:
    * Added raster saving function.
    * Added function to display raster values in a histogram.
    * Added thresholding functionality.
    * Added masking functionality.
* **Bug Fixes**:
    * Fixed inaccurate vegetation index calculation algorithms.

### Version 1.0 - August 06, 2025
* Initial release of the drone image processing program.
* Basic functionality for reading image bands.
* Basic functionality for calculating vegetation indices.