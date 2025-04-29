
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
