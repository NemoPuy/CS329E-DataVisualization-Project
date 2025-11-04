import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Create interactive time series comparison
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=[f"{model} - Predictions vs Actuals" for model in models_data.keys()],
    vertical_spacing=0.08,
    shared_xaxes=True
)

# Colors for consistency
colors = {'Actual': '#1f77b4', 'Linear Regression': '#ff7f0e',
          'Multiple Linear Regression': '#2ca02c', 'Random Forest': '#d62728'}

for idx, (model_name, (actual, predicted)) in enumerate(models_data.items(), 1):
    # Plot actual values
    fig.add_trace(
        go.Scatter(
            x=dates_test,
            y=actual,
            name='Actual' if idx == 1 else "",  # Show legend only once
            line=dict(color=colors['Actual'], width=2),
            opacity=0.8,
            hovertemplate='<b>Actual</b><br>Date: %{x}<br>Return: %{y:.4f}<extra></extra>'
        ),
        row=idx, col=1
    )

    # Plot predicted values
    fig.add_trace(
        go.Scatter(
            x=dates_test,
            y=predicted,
            name=model_name,
            line=dict(color=colors[model_name], width=1.5),
            opacity=0.8,
            hovertemplate=f'<b>{model_name}</b><br>Date: %{{x}}<br>Predicted: %{{y:.4f}}<extra></extra>'
        ),
        row=idx, col=1
    )

# Update layout
fig.update_layout(
    title_text="Time Series Prediction Comparison - All Models",
    height=900,
    showlegend=True,
    hovermode='x unified',
    template='plotly_white'
)

# Update y-axes labels
for i in range(1, 4):
    fig.update_yaxes(title_text="Returns", row=i, col=1)

fig.update_xaxes(title_text="Date", row=3, col=1)

# Add range slider for zooming
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    )
)

fig.show()