from app.result_model import AnalysisResult
from app.worker import WorkerHelper
from core.logic.utils import (
    generate_voronoi_poly, stack_bands
)
from core.logic.clip import clip_raster
from core.logic.transform import (
    prep_segmentation, segmentation, calc_vegetation_indices
)
from core.logic.mask import mask_stack, water_pred_mask
from core.logic.extract import extract_zonal_mean_pixels
from core.stats_calculator import (
    NutrientPlotCalculator, WaterPlotCalculator, PlantDiseaseCalculator
)
from core.classifier import (
    NutrientPlotClassifier, WaterPlotClassifier, PlantDiseaseClassifier
)

import logging
logger = logging.getLogger(__name__)
import os

class BaseController:
    _voronoi_cache = {}

    def __init__(self):
        self.task = None
        self.classifier = None
        self.stats_calc = None
        self.result = AnalysisResult()
        self.helper = None

    def _validate_raster(self, tif):
        import rasterio as rio
        with rio.open(tif) as src:
            if src.count < 7:
                raise IndexError(f"Raster requires a minimum of 7 bands (found {src.count}).")

    def prepare_mask_input(self, clipped_path, ndrei_path, out):
        return clipped_path

    def apply_mask(self, mask_input, segmented_path, ndvi_path, out):
        return mask_stack(mask_input, segmented_path, out)

    def prepare_classifier_input(self, masked_path, extracted_path, out):
        return extracted_path

    @classmethod
    def _get_cached_voronoi(cls, shp, out):
        logger.info("Accessing voronoi polygon...")
        cache_key = (shp, out)
        logger.info(cache_key)
        if cache_key in cls._voronoi_cache:
            path = cls._voronoi_cache[cache_key]
            if os.path.exists(path):
                return path

        shp_stem = os.path.splitext(os.path.basename(shp))[0] if shp else ""
        voronoi_path = os.path.join(out, f"{shp_stem}_voronoi.shp")

        if os.path.exists(voronoi_path):
            cls._voronoi_cache[cache_key] = voronoi_path
            return voronoi_path
        logger.warning("Voronoi polygon not found!")
        return None

    @classmethod
    def _cache_voronoi(cls, shp, out, path):
        if path:
            cls._voronoi_cache[(shp, out)] = path

    def get_voronoi_poly(self, shp, out):
        cached_path = self._get_cached_voronoi(shp, out)
        if cached_path:
            self.helper.progress(10, "Loading existing voronoi polygon...")
            logger.info(f"Using cached Voronoi polygon")
            return cached_path
        
        self.helper.progress(10, "Generating voronoi polygon...")
        poly_path = generate_voronoi_poly(
            shp,
            out,
            on_progress=self.helper.progress,
            subpoly_area=0.45
            )
        self._cache_voronoi(shp, out, poly_path)
        return poly_path

    def run(self, tif, shp, out, hooks=None):
        self.helper = WorkerHelper(hooks)
        self.helper.progress(5, "Checking raster...")
        self._validate_raster(tif)
        self.result.original_path = tif
        self.helper.check_cancel()

        multipolygon_path = None
        if shp:
            multipolygon_path = self.get_voronoi_poly(shp, out)
        self.helper.check_cancel()

        self.helper.progress(20, "Loading raster file...")
        clipped_path = clip_raster(tif, shp, out) if shp else tif
        self.helper.check_cancel()

        self.helper.progress(30, "Transforming with NDVI...")
        ndrei_path, ndvi_path = prep_segmentation(clipped_path, out)
        mask_input = self.prepare_mask_input(clipped_path, ndrei_path, out)
        self.helper.check_cancel()

        self.helper.progress(40, "Separating vegetation...")
        segmented_path = segmentation(
            clipped_path,
            ndvi_path,
            out,
            check_cancel=self.helper.cancelled,
            on_progress=self.helper.progress
        )
        self.helper.check_cancel()

        self.helper.progress(50, "Masking raster...")
        masked_path = self.apply_mask(mask_input, segmented_path, ndvi_path, out)
        self.helper.check_cancel()

        self.helper.progress(60, "Extracting pixel...")
        extracted_path = extract_zonal_mean_pixels(
            multipolygon_path,
            masked_path,
            out
        )
        self.helper.check_cancel()

        input_data = self.prepare_classifier_input(masked_path, extracted_path, out)

        self.helper.progress(70, f"Detecting {self.task}...")
        classified_path = self.classifier.run(
            input_path=input_data,
            shp_path=multipolygon_path,
            output_folder=out,
            check_cancel=self.helper.cancelled,
            on_progress=self.helper.progress
        )
        self.helper.check_cancel()

        self.helper.progress(90, "Calculating stats...")
        self.stats_calc.run(classified_path)
        self.helper.check_cancel()

        self.helper.progress(95, "Saving results...")
        self.result.clip_path = clipped_path
        self.result.transform_path = ndvi_path
        self.result.segmentation_path = segmented_path
        self.result.mask_path = masked_path
        self.result.extraction_path = extracted_path
        self.result.prediction_path = classified_path
        self.helper.progress(100, "Done")
        logger.info("All done")
        return self.result


class NutrientController(BaseController):
    def __init__(self):
        super().__init__()
        self.task = "nutrient"
        self.classifier = NutrientPlotClassifier()
        self.stats_calc = NutrientPlotCalculator()


class WaterController(BaseController):
    def __init__(self):
        super().__init__()
        self.task = "water"
        self.classifier = WaterPlotClassifier()
        self.stats_calc = WaterPlotCalculator()

    def apply_mask(self, mask_input, segmented_path, ndvi_path, out):
        return water_pred_mask(mask_input, ndvi_path, out)

    def prepare_classifier_input(self, masked_path, extracted_path, out):
        return calc_vegetation_indices(extracted_path, out)


class DiseaseController(BaseController):
    def __init__(self):
        super().__init__()
        self.task = "disease"
        self.classifier = PlantDiseaseClassifier()
        self.stats_calc = PlantDiseaseCalculator()

    def prepare_mask_input(self, clipped_path, ndrei_path, out):
        return stack_bands(
            lst_fitur=[clipped_path, ndrei_path],
            output_folder=out,
            output_filename="stack result.tif"
        )

    def prepare_classifier_input(self, masked_path, extracted_path, out):
        return masked_path
    