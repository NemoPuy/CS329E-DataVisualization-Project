# 5_visualization_generation.py
import altair as alt
import pandas as pd
import numpy as np

# Set this once when the module is imported
alt.data_transformers.enable('default', max_rows=None)


def create_prediction_chart(actual_values, predicted_values, dates=None, chart_title="Prediction Results"):
    import numpy as np
    import pandas as pd
    import altair as alt

    # Convert inputs
    y_clean = np.array(actual_values).copy()
    y_pred_clean = np.array(predicted_values).copy()

    if dates is not None and len(dates) != len(y_clean):
        raise ValueError("Length of dates must match actual and predicted values.")

    print(f"📊 Actual range: {y_clean.min():.6f} to {y_clean.max():.6f}")
    print(f"📊 Predicted range: {y_pred_clean.min():.6f} to {y_pred_clean.max():.6f}")

    # Build base DataFrame
    plot_data = pd.DataFrame({
        'time_index': range(len(y_clean)),
        'actual': y_clean,
        'predicted': y_pred_clean
    })
    if dates is not None:
        plot_data['date'] = pd.to_datetime(dates)

    melted_data = pd.melt(
        plot_data,
        id_vars=['time_index', 'date'] if 'date' in plot_data.columns else ['time_index'],
        value_vars=['actual', 'predicted'],
        var_name='series_type',
        value_name='value'
    )

    # Choose proper x field
    x_field = 'date:T' if 'date' in melted_data.columns else 'time_index:Q'
    hover_field = 'date' if 'date' in melted_data.columns else 'time_index'

    # --- Selections ---
    brush = alt.selection_interval(encodings=['x'])  # Altair 5
    hover = alt.selection_point(
        fields=[hover_field],
        nearest=True,
        on='mouseover',
        empty=False,
        clear='mouseout'
    )

    # --- Base chart ---
    base = (
        alt.Chart(melted_data)
        .encode(
            x=alt.X(x_field, title=None),
            y=alt.Y('value:Q', title='Returns', scale=alt.Scale(zero=False)),
            color=alt.Color(
                'series_type:N',
                title='Series',
                scale=alt.Scale(domain=['actual', 'predicted'],
                                range=['#1f77b4', '#ff7f0e'])
            )
        )
    )

    # --- Lines ---
    lines = base.mark_line().encode(opacity=alt.value(0.9))

    # --- Vertical hover rule ---
    rule = (
        alt.Chart(melted_data)
        .mark_rule(color='gray', strokeWidth=1)
        .encode(x=x_field)
        .transform_filter(hover)
    )

    # --- Text labels (actual + predicted values) ---
    text = (
        alt.Chart(melted_data)
        .mark_text(align='left', dx=5, dy=-5)
        .encode(
            x=x_field,
            text=alt.condition(
                hover,
                alt.Text('value:Q', format='.4f'),
                alt.value('')
            ),
            color='series_type:N'
        )
        .transform_filter(hover)
    )

    # --- Upper interactive chart ---
    upper = (
        alt.layer(lines, rule, text)
        .properties(width=800, height=350)
        .add_params(hover)
        .transform_filter(brush)
    )

    # --- Lower slider (overview) ---
    lower = (
        alt.Chart(melted_data)
        .mark_line()
        .encode(
            x=alt.X(x_field, title='Time Range'),
            y=alt.Y('value:Q', title=None, scale=alt.Scale(zero=False)),
            color=alt.Color('series_type:N', legend=None,
                            scale=alt.Scale(domain=['actual', 'predicted'],
                                            range=['#1f77b4', '#ff7f0e']))
        )
        .properties(width=800, height=100)
        .add_params(brush)
    )

    # --- Final composition ---
    chart = (
        alt.vconcat(upper, lower)
        .properties(title=chart_title)
        .configure_title(fontSize=16, anchor='start')
    )

    return chart