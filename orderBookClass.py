import pandas as pd
import numpy as np

class OBData:
    def __init__(self, historicalData):
        self.OBData_ = np.array(historicalData)
        self.__class__.step = 0
        self.__class__.OBIndex = {"bids":1, "bids_v":2, "asks":3, "asks_v":4, "transactionTime":5, "eventTime":6}

    
