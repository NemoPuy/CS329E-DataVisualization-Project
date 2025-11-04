import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# import master df
master_df = pd.read_csv('master_df.csv')

# calculate the
master_df['target_future_return'] = master_df['SPY_Close'].pct_change().shift(-1)

# create a new df called df_merge to take time, y_actual, y_pred's
df_merge = pd.DataFrame()
df_merge[['Date', 'y_actual']] = master_df[['Date', 'target_future_return']]

# import csv datasets of results from ML models
model_lin_reg = pd.read_csv('lin_reg_momentum_only_results.csv')
model_lin_reg = model_lin_reg.rename(columns = {'predicted_return': 'y_pred_lin_reg'})

model_multi_lin_reg = pd.read_csv('multi_lin_reg_with_momentum_and_VIX.csv')
model_multi_lin_reg = model_multi_lin_reg.rename(columns = {'predicted_return': 'y_pred_multi_lin_reg'})

model_random_forest = pd.read_csv('random_forest_results.csv')
model_random_forest = model_random_forest.rename(columns = {'predicted_return': 'y_pred_random_forest'})

# merge the df_merge with the model results
df_merge = pd.merge(
    left = df_merge,
    right = model_lin_reg[['Date', 'y_pred_lin_reg']],
    on = 'Date',
    how = 'left'
)
df_merge = pd.merge(
    left = df_merge,
    right = model_multi_lin_reg[['Date', 'y_pred_multi_lin_reg']],
    on = 'Date',
    how = 'left'
)
df_merge = pd.merge(
    left = df_merge,
    right = model_random_forest[['Date', 'y_pred_random_forest']],
    on = 'Date',
    how = 'left'
)

# make sure the Date is in datetime format
df_merge['Date'] = pd.to_datetime(df_merge['Date'], format='%Y-%m-%d')

# copy
df = df_merge.copy()
df['Date'] = pd.to_datetime(df['Date'])

# Create the plot
fig = go.Figure()

# Add real values
fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['y_actual'],
    mode='lines',
    name='Real Values',
    line=dict(color='grey', width=2),
    opacity=0.8
))

# Add Model 1 predictions
fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['y_pred_lin_reg'],
    mode='lines',
    name='Model 1',
    line=dict(color='blue', width=1.5),
    opacity=0.7
))

# Add Model 2 predictions
fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['y_pred_multi_lin_reg'],
    mode='lines',
    name='Model 2',
    line=dict(color='red', width=1.5),
    opacity=0.7
))

# Add Model 3 predictions
fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['y_pred_random_forest'],
    mode='lines',
    name='Model 3',
    line=dict(color='green', width=1.5),
    opacity=0.7
))

# Update layout
fig.update_layout(
    title='ML Model Predictions vs Real Values',
    xaxis_title='Date',
    yaxis_title='Values',
    hovermode='x unified',
    showlegend=True,
    width=1000,
    height=600,
    template='plotly_white'
)

# Add overview (slider) and range selector buttons
fig.update_xaxes(
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    ),
    rangeslider=dict(
        visible=True,
        thickness=0.1,
        bgcolor="lightgrey"
    ),
    type="date"
)

# Enable interactive legend behavior
fig.update_layout(
    legend=dict(
        title="Models",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    legend_itemclick="toggle",
    legend_itemdoubleclick="toggleothers"
)

# Show updated plot with range selector and overview slider
fig.show()

# Optional: Save as HTML
fig.write_html("time_series_prediction_comparison.html")
