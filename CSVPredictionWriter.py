import numpy as np
import pandas as pd
from RatePredictor import RatePredictor
class CSVPredicitionWriter:
    def __init__(self,predictor,target_file = 'target.csv'):
        self.predictor = predictor
        self.target_file = target_file

