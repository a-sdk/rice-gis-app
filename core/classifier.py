from core.logic.predict import (
    predict_disease, 
    deteksi_penyakit_petak,
    predict_water_plot,
    predict_nutrient_plont
)
from path_config import ModelRegistry
import logging
logger = logging.getLogger(__name__)

class BaseClassifier:
    MODEL_KEY_1 = None
    MODEL_KEY_2 = None
    MODEL_KEY_3 = None
    SCALER_KEY_1 = None
    SCALER_KEY_2 = None
    SCALER_KEY_3 = None

    def __init__(self):
        self.scaler_1 = None
        self.scaler_2 = None
        self.scaler_3 = None
        self.model_1 = None
        self.model_2 = None
        self.model_3 = None
        self.result = None

    def _load_model(self):
        if self.MODEL_KEY_1 is None or self.SCALER_KEY_1 is None:
            raise ValueError("Path model belum ditentukan di child class!")
        if self.model_1 is not None:
            return
        
        import joblib
        with joblib.parallel_backend('threading'):
            if self.SCALER_KEY_1 is not None:
                self.scaler_1 = joblib.load(str(ModelRegistry.scaler_path(self.SCALER_KEY_1)))
                
            if self.SCALER_KEY_2 is not None:
                self.scaler_2 = joblib.load(str(ModelRegistry.scaler_path(self.SCALER_KEY_2)))
                
            if self.SCALER_KEY_3 is not None:
                self.scaler_3 = joblib.load(str(ModelRegistry.scaler_path(self.SCALER_KEY_3)))


        if self.MODEL_KEY_1 is not None:
            self.model_1 = str(ModelRegistry.model_path(self.MODEL_KEY_1))
            
        if self.MODEL_KEY_2 is not None:
            self.model_2 = str(ModelRegistry.model_path(self.MODEL_KEY_2))
            
        if self.MODEL_KEY_3 is not None:
            self.model_3 = str(ModelRegistry.model_path(self.MODEL_KEY_3))

    def run(self, input_path, output_folder, shp_path=None, check_cancel=None, on_progress=None):
        logger.info(f"Memulai prediksi dengan {self.__class__.__name__}...")
        try:
            self.result = self._do_prediction(input_path, output_folder, shp_path, check_cancel, on_progress)
            return self.result
        except Exception as e:
            logger.error(f"ERROR: {type(e).__name__}: {e}", exc_info=True)
            return None  
           
    def _do_prediction(self, *args, **kwargs):
            raise NotImplementedError("Child class harus mengimplementasikan method _do_prediction")
    
class PlantDiseaseClassifier(BaseClassifier):
    """
    Kelas untuk deteksi penyakit per rumpun.
    """ 
    MODEL_KEY_1 = "disease_detection"
    SCALER_KEY_1 = "disease_scaler"
    def __init__(self):
        super().__init__()

    def _do_prediction(self, input_path, output_folder, shp_path, check_cancel, on_progress):
        self._load_model()
        output_path = predict_disease(
            scaler=self.scaler_1, 
            model=self.model_1, 
            input_path=input_path, 
            output_folder=output_folder, 
            check_cancel=check_cancel, 
            on_progress=on_progress
            )
        return output_path
    
class DiseasePlotClassifier(BaseClassifier):
    """
    Kelas untuk deteksi penyakit per plot/petak.
    """ 
    MODEL_KEY_1 = "disease_detection"
    SCALER_KEY_1 = "disease_scaler"
    def __init__(self):
        super().__init__()

    def _do_prediction(self, input_path, output_folder, shp_path, check_cancel, on_progress):
        self._load_model()
        output_gpkg = deteksi_penyakit_petak(
            self.scaler_1, 
            self.model_1, 
            input_path, 
            shp_path, 
            output_folder, 
            check_cancel,
            on_progress
            )
        return output_gpkg
    
class WaterPlotClassifier(BaseClassifier):
    """
    Kelas untuk deteksi ketersedian air per plot/petak.
    """ 
    MODEL_KEY_1 = "water_availability"
    SCALER_KEY_1 = "water_scaler"
    def __init__(self):
        super().__init__()

    def _do_prediction(self, input_path, output_folder, shp_path, check_cancel, on_progress):
        self._load_model()
        output_gpkg = predict_water_plot(
            scaler=self.scaler_1, 
            model_reg=self.model_1, 
            input_path=input_path, 
            shp_path=shp_path, 
            output_folder=output_folder, 
            check_cancel=check_cancel,
            on_progress=on_progress
            )
        return output_gpkg
    
class NutrientPlotClassifier(BaseClassifier):
    """
    Kelas untuk deteksi ketersediaan nitrogen per plot/petak.
    """ 
    MODEL_KEY_1 = "nitrogen_availability"
    MODEL_KEY_2 = "phospor_availability"
    MODEL_KEY_3 = "kalium_availability"
    SCALER_KEY_1 = "nitrogen_scaler"

    def __init__(self):
        super().__init__()

    def _do_prediction(self, input_path, output_folder, shp_path, check_cancel, on_progress):
        self._load_model()
        output_gpkg = predict_nutrient_plont(
            scaler_n=self.scaler_1, 
            scaler_p=self.scaler_1,
            scaler_k=self.scaler_1,
            model_n=self.model_1,
            model_p=self.model_2, 
            model_k=self.model_3,
            input_path=input_path, 
            shp_path=shp_path, 
            output_folder=output_folder, 
            check_cancel=check_cancel,
            on_progress=on_progress
            )
        return output_gpkg