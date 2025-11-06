# import libraries
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


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

# Create Volatility Regime Labels (based on VIX)
# Convert Date to datetime and align
master_df['Date'] = pd.to_datetime(master_df['Date'])
df_merge['Date'] = pd.to_datetime(df_merge['Date'])

# Clean VIX and smooth slightly (optional)
master_df['VIX_SMA_5'] = master_df['VIX'].rolling(5).mean()

# Define thresholds (quantile-based gives more balance than median split)
low_th = master_df['VIX_SMA_5'].quantile(0.33)
high_th = master_df['VIX_SMA_5'].quantile(0.67)

def classify_vol(vix):
    if vix <= low_th:
        return 'Low Vol'
    elif vix <= high_th:
        return 'Medium Vol'
    else:
        return 'High Vol'

master_df['Vol_Regime'] = master_df['VIX_SMA_5'].apply(classify_vol)

# Merge regime labels into df_merge
df_merge = df_merge.merge(master_df[['Date', 'Vol_Regime']], on='Date', how='left')

# Compute Model Errors
# Signed errors (predicted - actual)
df_merge['error_lin_reg'] = df_merge['y_pred_lin_reg'] - df_merge['y_actual']
df_merge['error_multi_lin_reg'] = df_merge['y_pred_multi_lin_reg'] - df_merge['y_actual']
df_merge['error_random_forest'] = df_merge['y_pred_random_forest'] - df_merge['y_actual']

# Absolute errors (for performance magnitude)
df_merge['abs_error_lin_reg'] = df_merge['error_lin_reg'].abs()
df_merge['abs_error_multi_lin_reg'] = df_merge['error_multi_lin_reg'].abs()
df_merge['abs_error_random_forest'] = df_merge['error_random_forest'].abs()

# Melt the error columns into long format for plotting
df_melt = df_merge.melt(
    id_vars=['Date', 'Vol_Regime'],
    value_vars=[
        'abs_error_lin_reg',
        'abs_error_multi_lin_reg',
        'abs_error_random_forest'
    ],
    var_name='Model',
    value_name='Absolute Error'
)

# Clean up model names for readability
df_melt['Model'] = df_melt['Model'].replace({
    'abs_error_lin_reg': 'Linear Regression',
    'abs_error_multi_lin_reg': 'Multi Linear Regression',
    'abs_error_random_forest': 'Random Forest'
})

# Create a baseline violin plot
fig = px.violin(
    df_melt,
    x='Vol_Regime',
    y='Absolute Error',
    color='Model',
    box=True,           # shows embedded boxplot for summary
    points='all',       # show all data points
    hover_data=['Date'],# adds more context on hover
    title='Model Error Distribution by Volatility Regime',
    template='plotly_white'
)

# Customize layout for readability
fig.update_layout(
    legend_title_text='Model',
    xaxis_title='Volatility Regime',
    yaxis_title='Absolute Prediction Error',
    width=950,
    height=600,
    font=dict(size=12),
    title_font=dict(size=18)
)

# Show the figure
fig.show()


