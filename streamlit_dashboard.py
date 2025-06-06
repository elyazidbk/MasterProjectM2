from data_pipeline import data_extractor, extract_json_blocks, get_position, log_extractor, safe_convert
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
st.set_page_config(layout="wide")
import re


# =============================================================================
#                          --- Data Extraction ---
# =============================================================================


log_file = r"/Users/leo/Desktop/Prosperity3/backtesterLogs/98dc1678-6056-4a36-861f-467e90e8a154.log"


sandbox_part, activities_logs, trade_history_json = data_extractor(log_file)
sandbox_logs = extract_json_blocks(sandbox_part)


md = pd.DataFrame(activities_logs)
md["timestamp"] = md["timestamp"].apply(lambda x: int(x))
md["profit_and_loss"] = md["profit_and_loss"].apply(lambda x: float(x))

try:
    md["realized_pnl"] = md["realized_pnl"].apply(lambda x: float(x))
except:
    print("no realized_pnl column in data from IMC results")
    pass

for i in range(1,4):
    md[f"bid_price_{i}"] = md[f"bid_price_{i}"].apply(safe_convert)
    md[f"ask_price_{i}"] = md[f"ask_price_{i}"].apply(safe_convert)
    md[f"bid_volume_{i}"] = md[f"bid_volume_{i}"].apply(safe_convert)
    md[f"ask_volume_{i}"] = md[f"ask_volume_{i}"].apply(safe_convert)

PRODUCTS = md["product"].unique()

md.to_csv("round4_md.csv")

df_trades = pd.DataFrame(trade_history_json)
df_trades["timestamp"] = df_trades["timestamp"].apply(lambda x: int(x))
df_own_trades = df_trades[(df_trades["buyer"] == "SUBMISSION") | (df_trades["seller"] == "SUBMISSION")]
df_market_trades = df_trades[(df_trades["buyer"] == "") & (df_trades["seller"] == "")]

df_market_trades.to_csv("round4_trades.csv")

md.to_csv("inSampleMd_round3.csv")
df_market_trades.to_csv("inSampleTrades_round3.csv")
        
df_logs = log_extractor(sandbox_logs)
df_position = get_position(df_logs).dropna()
df_position["timestamp"] = df_position["timestamp"].apply(lambda x: int(x))

for product in PRODUCTS:
    if product+"_pos" not in df_position.columns:
        df_position[product+"_pos"] = 0

total_pnl = pd.DataFrame({"timestamp": md["timestamp"].unique(),"total_pnl": np.zeros(len(md["timestamp"].unique()))})

# Streamlit multi-select to select multiple assets
asset_options = df_trades['symbol'].unique()
selected_assets = st.multiselect('Select Assets', asset_options)

for asset in asset_options:
    selected_pnl = md[md['product'] == asset]["profit_and_loss"]
    total_pnl["total_pnl"] += selected_pnl.values

# =============================================================================
#                          End data extraction
# =============================================================================



# Initialize the figure
# Create subplots, sharing x-axes between two columns
st.subheader("Execution Dashboard")

fig = make_subplots(
    rows=2, cols=2,  # Two rows, two columns
    shared_xaxes=True,  # Share x-axis between columns 1 and 2
    vertical_spacing=0.05,  # Increased space between the two charts (higher than before)
    horizontal_spacing=0.15,  # Increased space between the columns (higher than before)
    subplot_titles=("Execution", "Markout PnL per Asset", "Inventory", "Realized PnL per Asset"),
    row_heights=[0.5, 0.5],  # Adjust row height proportions to allocate more space for price chart
    specs=[[{"secondary_y": True}, {"secondary_y": True}], 
           [{"secondary_y": True}, {"secondary_y": True}]]  # Secondary axis for the PnL plot
)
fig.update_xaxes(matches='x')
# Add traces for profit_and_loss for each symbol in the second plot (PnL chart)
color_palette = px.colors.qualitative.Plotly
# Assign colors from the palette (cycling if more assets than colors)
asset_colors = {}


# Loop through the selected assets and plot their data
for idx, selected_asset in enumerate(selected_assets):
    
    # Price data for the selected asset (using the 'md' dataframe)
    selected_prices = md[md['product'] == selected_asset]
    # Trades data for the selected asset (using the 'df_trades' dataframe)
    selected_trades = df_trades[df_trades['symbol'] == selected_asset]
    # Buy Orders
    buy_trades = selected_trades[selected_trades['buyer'] == "SUBMISSION"]
    # Sell Orders
    sell_trades = selected_trades[selected_trades['seller'] == "SUBMISSION"]
    # Market Orders
    market_trades = selected_trades[(selected_trades['buyer'] == "") & (selected_trades['seller'] == "")]
    # PnL
    selected_pnl = md[md['product'] == selected_asset]
    # Position
    selected_position = df_position[["timestamp", f"{selected_asset}_pos"]]

    bid_volumes = selected_prices[[f'bid_volume_{i}' for i in range(1, 4)]].fillna(0).values
    bid_prices  = selected_prices[[f'bid_price_{i}' for i in range(1, 4)]].fillna(0).values

    ask_volumes = selected_prices[[f'ask_volume_{i}' for i in range(1, 4)]].fillna(0).values
    ask_prices  = selected_prices[[f'ask_price_{i}'  for i in range(1, 4)]].fillna(0).values

    # 2️⃣ Compute VWAP Bid and VWAP Ask using vectorized numpy ops
    vwap_bid = np.sum(bid_volumes * bid_prices, axis=1) / np.sum(bid_volumes, axis=1)
    vwap_ask = np.sum(ask_volumes * ask_prices, axis=1) / np.sum(ask_volumes, axis=1)

    # 3️⃣ VWAP Mid is the average of VWAP bid and VWAP ask
    vwap_mid = (vwap_bid + vwap_ask) / 2
    vwap_ema = pd.Series(vwap_mid).ewm(span=1.8, adjust=False).mean()
   
    color = color_palette[idx % len(color_palette)]
    asset_colors[selected_asset] = color
    
    if idx == 0:
        # Plot bid price for the selected asset (primary y-axis)
        fig.add_trace(go.Scatter(x=selected_prices['timestamp'], y=selected_prices['bid_price_1'], 
                                mode='lines', name=f"{selected_asset} bid", line=dict(color='blue', width=2)),
                    row=1, col=1, secondary_y=False)
        
        # Plot ask price for the selected asset (primary y-axis)
        fig.add_trace(go.Scatter(x=selected_prices['timestamp'], y=selected_prices['ask_price_1'], 
                                mode='lines', name=f"{selected_asset} ask", line=dict(color='orange', width=2)),
                    row=1, col=1, secondary_y=False)
        

        
        fig.add_trace(go.Scatter(x=buy_trades['timestamp'], y=buy_trades['price'], 
                                mode='markers', name=f"{selected_asset} Buy Orders", 
                                marker=dict(symbol='triangle-up', color='green', size=10)),
                    row=1, col=1, secondary_y=False)
        
        
        fig.add_trace(go.Scatter(x=sell_trades['timestamp'], y=sell_trades['price'], 
                                mode='markers', name=f"{selected_asset} Sell Orders", 
                                marker=dict(symbol='triangle-down', color='red', size=10)),
                    row=1, col=1, secondary_y=False)
        
        
        fig.add_trace(go.Scatter(x=market_trades['timestamp'], y=market_trades['price'], 
                                mode='markers', name=f"{selected_asset} Market Orders", 
                                marker=dict(symbol='circle', color='white', size=8)),
                    row=1, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(x=selected_pnl["timestamp"], y=selected_pnl["profit_and_loss"],
                                mode='lines', name=f'{selected_asset} Markout PnL', line=dict(color=asset_colors[selected_asset], dash='dash')),
                    row=1, col=2, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=selected_prices['timestamp'],
            y=vwap_ema.values,
            mode='lines',
            name=f"EMA VWAP mid",
            visible='legendonly'))
        
        try:
            fig.add_trace(go.Scatter(x=selected_pnl["timestamp"], y=selected_pnl["realized_pnl"],
                                    mode='lines', name=f'{selected_asset} Realized PnL', line=dict(color=asset_colors[selected_asset], dash='dash')),
                        row=2, col=2, secondary_y=False)
        except:
            pass

        fig.add_trace(go.Scatter(x=selected_position['timestamp'], y=selected_position[f'{selected_asset}_pos'], 
                                mode='lines', name=f"{selected_asset} position", line=dict(color=asset_colors[selected_asset], width=2)),
                    row=2, col=1, secondary_y=False)
         
    else:
        # Plot bid price for the selected asset (primary y-axis)
        fig.add_trace(go.Scatter(x=selected_prices['timestamp'], y=selected_prices['bid_price_1'], 
                                mode='lines', name=f"{selected_asset} bid", line=dict(color='blue', width=2), visible='legendonly'),
                    row=1, col=1, secondary_y=True)
        
        # Plot ask price for the selected asset (primary y-axis)
        fig.add_trace(go.Scatter(x=selected_prices['timestamp'], y=selected_prices['ask_price_1'], 
                                mode='lines', name=f"{selected_asset} ask", line=dict(color='orange', width=2), visible='legendonly'),
                    row=1, col=1, secondary_y=True)
        

        
        fig.add_trace(go.Scatter(x=buy_trades['timestamp'], y=buy_trades['price'], 
                                mode='markers', name=f"{selected_asset} Buy Orders", 
                                marker=dict(symbol='triangle-up', color='green', size=10), visible='legendonly'),
                    row=1, col=1, secondary_y=True)
        
        
        fig.add_trace(go.Scatter(x=sell_trades['timestamp'], y=sell_trades['price'], 
                                mode='markers', name=f"{selected_asset} Sell Orders", 
                                marker=dict(symbol='triangle-down', color='red', size=10), visible='legendonly'),
                    row=1, col=1, secondary_y=True)
        
        
        fig.add_trace(go.Scatter(x=market_trades['timestamp'], y=market_trades['price'], 
                                mode='markers', name=f"{selected_asset} Market Orders", 
                                marker=dict(symbol='circle', color='white', size=8), visible='legendonly'),
                    row=1, col=1, secondary_y=True)    


        fig.add_trace(go.Scatter(x=selected_pnl["timestamp"], y=selected_pnl["profit_and_loss"],
                                mode='lines', name=f'{selected_asset} Markout PnL', line=dict(color=asset_colors[selected_asset], dash='dash')),
                    row=1, col=2, secondary_y=True)    
        try:
            fig.add_trace(go.Scatter(x=selected_pnl["timestamp"], y=selected_pnl["realized_pnl"],
                                    mode='lines', name=f'{selected_asset} Realized PnL', line=dict(color=asset_colors[selected_asset], dash='dash')),
                        row=2, col=2, secondary_y=True)
        except:
            pass
        

        fig.add_trace(go.Scatter(x=selected_position['timestamp'], y=selected_position[f'{selected_asset}_pos'], 
                                mode='lines', name=f"{selected_asset} position", line=dict(color=asset_colors[selected_asset], width=2)),
                    row=2, col=1, secondary_y=True)
    

fig.add_trace(go.Scatter(x=total_pnl["timestamp"], y=total_pnl["total_pnl"],
                        mode='lines', name=f'Total PnL', line=dict(color="skyblue")),
            row=1, col=2, secondary_y=False)


    

# Update layout for the chart with multiple y-axes
fig.update_layout(
    title='Price and Trades for Selected Assets',
    xaxis3_title='Timestamp',
    template='plotly_dark',
    showlegend=True,
    height=1300,  # Adjust height for clarity
    width=2000,  # Set the width manually in pixels
    margin=dict(t=80, b=0, l=40, r=40),  # Adjust margins for better spacing
    
    # Adjusting axis tick font size
    xaxis3=dict(
        tickfont=dict(size=20),  # Larger font for x-axis values
    ),
    xaxis4=dict(
        tickfont=dict(size=20),  # Larger font for x-axis values
    ),
    yaxis=dict(
        tickfont=dict(size=20),  # Larger font for y-axis values
    ),
    yaxis2=dict(
        tickfont=dict(size=20),  # Larger font for y-axis values in PnL chart
    ),
    yaxis3=dict(
        tickfont=dict(size=20),  # Larger font for y-axis values in PnL chart
    ),
    yaxis4=dict(
        tickfont=dict(size=20),  # Larger font for y-axis values in PnL chart
    ),
    yaxis5=dict(
        tickfont=dict(size=20),  # Larger font for y-axis values in PnL chart
    ),
    yaxis6=dict(
        tickfont=dict(size=20),  # Larger font for y-axis values in PnL chart
    ),
    yaxis7=dict(
        tickfont=dict(size=20),  # Larger font for y-axis values in PnL chart
    ),
    yaxis8=dict(
        tickfont=dict(size=20),  # Larger font for y-axis values in PnL chart
    )

)

fig.update_yaxes(
    title_text="Price", row=1, col=1,
    showgrid=True,  # Keep gridlines visible, or set to False to hide them
    showline=True,  # Adds a line at the edge of the y-axis
    linewidth=1,  # Line width for the y-axis
    ticks="outside",  # Ticks outside the plot for better separation
    ticklen=5,  # Length of the ticks
    tickwidth=1,  # Thickness of the ticks
    tickformat=".2f",  # Use 2 decimal places for PnL values
    tickangle=0  # Set tick angle to 0 to avoid angle issues with long tick values
)

fig.update_yaxes(
    title_text="PnL for Each Asset", 
    row=1, col=2,
    showgrid=True,  # Keep gridlines visible, or set to False to hide them
    showline=True,  # Adds a line at the edge of the y-axis
    linewidth=1,  # Line width for the y-axis
    ticks="outside",  # Ticks outside the plot for better separation
    ticklen=5,  # Length of the ticks
    tickwidth=1,  # Thickness of the ticks
    tickformat=".2f",  # Use 2 decimal places for PnL values
    tickangle=0  # Set tick angle to 0 to avoid angle issues with long tick values
)

fig.update_yaxes(
    title_text="Inventory", 
    row=2, col=1,
    showgrid=True,  # Keep gridlines visible, or set to False to hide them
    showline=True,  # Adds a line at the edge of the y-axis
    linewidth=1,  # Line width for the y-axis
    ticks="outside",  # Ticks outside the plot for better separation
    ticklen=5,  # Length of the ticks
    tickwidth=1,  # Thickness of the ticks
    tickformat=".2f",  # Use 2 decimal places for PnL values
    tickangle=0  # Set tick angle to 0 to avoid angle issues with long tick values
)

fig.update_yaxes(
    title_text="Markout PnL per asset", 
    row=2, col=2,
    showgrid=True,  # Keep gridlines visible, or set to False to hide them
    showline=True,  # Adds a line at the edge of the y-axis
    linewidth=1,  # Line width for the y-axis
    ticks="outside",  # Ticks outside the plot for better separation
    ticklen=5,  # Length of the ticks
    tickwidth=1,  # Thickness of the ticks
    tickformat=".2f",  # Use 2 decimal places for PnL values
    tickangle=0  # Set tick angle to 0 to avoid angle issues with long tick values
)

st.plotly_chart(fig, use_container_width=True)

# Fourth: DataFrame of Order Book at Selected Timestamp
st.subheader("Order Book at Selected Timestamp")


# Set the slider min and max to the range of timestamps
selected_timestamp = st.slider("Select Timestamp", min_value=int(df_position['timestamp'].min()), max_value=int(df_position['timestamp'].max()), step=100)

@st.fragment
def display_OBdata_at_timestamp(timestamp):
    # Filter the order book and market trades at the selected timestamp
    order_book_at_timestamp = md[md['timestamp'] == timestamp]
    trades_at_timestamps = df_trades[df_trades["timestamp"] == timestamp]
    
    # Streamlit fragment to display data

    st.dataframe(order_book_at_timestamp, use_container_width=True, height=500)  # Making it scrollable and responsiv
    st.dataframe(trades_at_timestamps, use_container_width=True)  # Making it scrollable and responsive

