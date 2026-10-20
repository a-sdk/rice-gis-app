from core.logic.utils import calculate_cluster_distribution, calculate_plot_distribution
from path_config import InfoRegistry
import logging
logger = logging.getLogger(__name__)

class StatsCalculator:
    """
    Kelas untuk menghitung sebaran per plot/petak.
    """
    def __init__(self):
        self.result = None
        self.legend = None

    def get_legend(self, name):
        raw_legend = InfoRegistry.get_legend(name)
        self.legend = {int(k): v for k, v in raw_legend.items()}

    def run(self, input_path):
        logger.info("Menghitung sebaran dan memunculkan hasil...") 
        try:
            self.result = self._do_calculation(input_path)
            return self.result
        except Exception as e:
            logger.error(f"ERROR: {type(e).__name__}: {e}", exc_info=True)
            return None 
        
    def _do_calculation(self, *args, **kwargs):
        raise NotImplementedError("Child class harus mengimplementasikan method _do_prediction")
        
class PlantDiseaseCalculator(StatsCalculator):
    """
    Kelas untuk menghitung sebaran per rumpun.
    """ 
    def __init__(self):
        super().__init__()
        self.get_legend("disease")

    def _do_calculation(self, input_path):
        calculate_cluster_distribution(input_path, legend_dict=self.legend)
        
class DiseasePlotCalculator(StatsCalculator):
    """
    Kelas untuk menghitung sebaran per plot/petak.
    """ 
    def __init__(self):
        super().__init__()
        self.get_legend("disease")

    def _do_calculation(self, input_path):
        calculate_plot_distribution(input_path, legend_dict=self.legend)
    
class NutrientPlotCalculator(StatsCalculator):
    """
    Kelas untuk menghitung sebaran per plot/petak.
    """ 
    def __init__(self):
        super().__init__()
        self.get_legend("nutrient")

    def _do_calculation(self, input_path):
        calculate_plot_distribution(input_path, legend_dict=self.legend)

class WaterPlotCalculator(StatsCalculator):
    """
    Kelas untuk menghitung sebaran per plot/petak.
    """ 
    def __init__(self):
        super().__init__()
        self.get_legend("water")

    def _do_calculation(self, input_path):
        calculate_plot_distribution(input_path, legend_dict=self.legend)
