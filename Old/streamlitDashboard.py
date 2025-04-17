from strats.basicStrat import basicStrat
from strats.movingAverageStrat import movingAverageStrat
from strats.rsiStrat import rsiStrat

from backtesterClass.orderClass import orders
from backtesterClass.orderBookClass import OBData
from backtesterClass.analysisClass import analysisClass

import numpy as np
import pandas as pd
from tqdm import tqdm
from BacktesterOHLCV.debug import logger
import streamlit as st

from cProfile import Profile
from pstats import SortKey, Stats
import gc

# Improve computationnal performance of the backtester - increases garbage collector threshold

allocs, gen1, gen2 = gc.get_threshold()
allocs = 5000
gen1 = gen1*2
gen2=gen2*2
gc.set_threshold(allocs, gen1,gen2)


dataClass = OBData(pd.read_csv(r'/Users/leo/Downloads/BTCUSD_241227-bookTicker-2024-09-01.csv', sep=","))

autoTrader = basicStrat("autoTrader")
momentumTrader = movingAverageStrat("momentumTrader", short_window = 1000, long_window=10000)
rsiTrader = rsiStrat("rsiTrader", window=1000, buyThreshold=30, sellThreshold=70, alpha=0.002)

for _ in tqdm(range(len(dataClass.OBData_))):
    orderClass = orders(dataClass.OBData_[OBData.step])
    autoTrader.strategy(orderClass)
    momentumTrader.strategy(orderClass)
    rsiTrader.strategy(orderClass)   
    OBData.step +=1 

analysisBasic = analysisClass(autoTrader)
analysisMovingAverage = analysisClass(momentumTrader)
analysisRSI = analysisClass(rsiTrader)

dashboardBasic = analysisBasic.create_dashboard()
logger.info(f"dashboardBasic done!")
dashboardMovingAverage = analysisMovingAverage.create_dashboard()
logger.info(f"dashboardMovingAverage done!")
dashboardRSI = analysisRSI.create_dashboard()
logger.info(f"dashboardRSI done!")


# st.title('Trading Dashboard')

# Define available strategies
strategies = {
    "Basic Strategy": dashboardBasic,
    "Moving Average Strategy": dashboardMovingAverage,
    "RSI Strategy" : dashboardRSI   
            }


# Dropdown for strategy selection
@st.fragment
def rerun():
    st.empty()
    selected_strategy = st.selectbox("Select an Auto Trader Strategy", list(strategies.keys()))
    selected_dashboard = strategies[selected_strategy]
    analysisClass.streamlitDashboard(selected_dashboard)

rerun()   
# st.plotly_chart(selected_dashboard)