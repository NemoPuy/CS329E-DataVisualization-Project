# viz_utils.py
import altair as alt
import pandas as pd
import numpy as np

# Set this once when the module is imported
alt.data_transformers.enable('default', max_rows=None)


def create_prediction_chart(actual_values, predicted_values, dates=None, chart_title="Prediction Results"):
    """
    Create a faceted chart with actual and predicted in separate panels.
    """
    # Convert to numpy arrays
    y_clean = np.array(actual_values).copy()
    y_pred_clean = np.array(predicted_values).copy()

    print(f"📊 Actual range: {y_clean.min():.6f} to {y_clean.max():.6f}")
    print(f"📊 Predicted range: {y_pred_clean.min():.6f} to {y_pred_clean.max():.6f}")

    # Prepare data
    time_index = range(len(y_clean))

    plot_data = pd.DataFrame({
        'time_index': time_index,
        'actual': y_clean,
        'predicted': y_pred_clean
    })

    if dates is not None:
        plot_data['date'] = dates

    # Melt for Altair - create separate rows for actual vs predicted
    melted_data = pd.melt(
        plot_data,
        id_vars=['time_index', 'date'] if 'date' in plot_data.columns else ['time_index'],
        value_vars=['actual', 'predicted'],
        var_name='series_type',
        value_name='value'
    )

    # Create faceted chart
    base = alt.Chart(melted_data).mark_line().encode(
        x=alt.X('time_index:Q', title='Time Index'),
        y=alt.Y('value:Q', title='Returns', scale=alt.Scale(zero=False)),
        color=alt.Color('series_type:N',
                        scale=alt.Scale(domain=['actual', 'predicted'],
                                        range=['#1f77b4', '#ff7f0e'])),
        tooltip=['series_type', 'value', 'date'] if 'date' in melted_data.columns
        else ['series_type', 'value']
    ).properties(
        width=800,
        height=200
    )

    # Facet by series_type
    faceted_chart = base.facet(
        facet=alt.Facet('series_type:N', title=None),
        columns=1
    ).properties(
        title=chart_title
    )

    return faceted_chart


def create_prediction_error_chart(actual_values, predicted_values, dates=None):
    """Create a chart showing prediction errors (actual - predicted)"""
    errors = actual_values - predicted_values

    plot_data = pd.DataFrame({
        'time_index': range(len(errors)),
        'prediction_error': errors
    })

    if dates is not None:
        plot_data['date'] = dates

    chart = alt.Chart(plot_data).mark_line(color='red').encode(
        x=alt.X('time_index:Q', title='Time Index'),
        y=alt.Y('prediction_error:Q', title='Prediction Error (Actual - Predicted)'),
        tooltip=['prediction_error', 'date'] if 'date' in plot_data.columns else ['prediction_error']
    ).properties(
        width=800,
        height=400,
        title="Prediction Errors Over Time"
    )

    # Add zero reference line
    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='black', strokeDash=[5, 5]).encode(y='y:Q')

    return chart + zero_line


def create_residuals_chart(actual_values, predicted_values, dates=None):
    """
    Create a residuals plot for model diagnostics.
    """
    residuals = actual_values - predicted_values

    plot_data = pd.DataFrame({
        'predicted': predicted_values,
        'residuals': residuals
    })

    if dates is not None:
        plot_data['date'] = dates

    chart = alt.Chart(plot_data).mark_circle(size=30, opacity=0.6).encode(
        x=alt.X('predicted:Q', title='Predicted Values'),
        y=alt.Y('residuals:Q', title='Residuals'),
        tooltip=['predicted', 'residuals', 'date'] if 'date' in plot_data.columns
        else ['predicted', 'residuals']
    ).properties(
        width=600,
        height=300,
        title='Residuals Plot'
    )

    # Add zero line
    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='red', strokeDash=[5, 5]).encode(y='y:Q')

    return chart + zero_line