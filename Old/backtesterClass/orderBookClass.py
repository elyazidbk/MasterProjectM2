import pandas as pd
import numpy as np

class OBData:
    def __init__(self, historicalData):
        self.__class__.OBData_ = np.array(historicalData)
        self.__class__.step = 0
        self.__class__.OBIndex = {"bids":1, "bids_v":2, "asks":3, "asks_v":4, "transactionTime":5, "eventTime":6}
    
    @classmethod
    def mid(self):
       mid_ = (self.OBData_[self.step][self.OBIndex["bids"]]+self.OBData_[self.step][self.OBIndex["asks"]])/2
       return mid_