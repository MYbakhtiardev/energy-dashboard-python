import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from prophet import Prophet

# Load and preprocess data
df = pd.read_csv('Energy_consumption.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df_clean = df.dropna()
df_clean['HVACUsage'] = (df_clean['HVACUsage'] == 'On').astype(int)
df_clean['LightingUsage'] = (df_clean['LightingUsage'] == 'On').astype(int)

X = df_clean[['Temperature', 'Humidity', 'SquareFootage', 'Occupancy',
              'RenewableEnergy', 'HVACUsage', 'LightingUsage']]
y = df_clean['EnergyConsumption']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Dash app
app = dash.Dash(__name__)

# Reusable card component
def graph_card(title, component, explanation):
    return html.Div([
        html.H3(title),
        component if isinstance(component, dcc.Graph) else dcc.Graph(figure=component),
        html.Div(explanation, style={'padding': '10px', 'fontSize': '16px', 'color': '#555', 'fontFamily': 'Georgia, serif'})
    ], style={
        'backgroundColor': '#f9f9f9',
        'border': '1px solid #ccc',
        'borderRadius': '10px',
        'padding': '15px',
        'marginBottom': '30px',
        'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
    })

app.layout = html.Div([
    html.H1("⚡ Energy Consumption Dashboard", style={'textAlign': 'center', 'paddingTop': '20px'}),

    html.Div([
        graph_card(
            "Energy Consumption Over Time",
            px.line(df, x='Timestamp', y='EnergyConsumption'),
            html.Div([
                html.P("This line graph shows energy consumption over time."),
                html.Ul([
                    html.Li("Peaks often occur during weekends or holidays with high occupancy.", style={'fontWeight': 'bold'}),
                    html.Li("Drops typically happen during off-peak hours like early mornings.", style={'fontWeight': 'bold'}),
                ])
            ])
        ),

        graph_card(
            "Feature Correlation Heatmap",
            px.imshow(df.select_dtypes(include=['float64', 'int64']).corr()),
            html.Div([
                html.P("Correlations with energy usage:"),
                html.Ul([
                    html.Li("✔️ Temperature & Energy: HVAC increases consumption in extreme temps.", style={'fontWeight': 'bold'}),
                    html.Li("✔️ Occupancy & Energy: More people → more lighting/HVAC/device usage.", style={'fontWeight': 'bold'}),
                    html.Li("➖ Renewable Energy: Negative correlation (reduces grid load).", style={'fontWeight': 'bold'}),
                ])
            ])
        ),

        graph_card(
            "Hourly Energy Consumption Distribution",
            px.box(df, x=df['Timestamp'].dt.hour, y='EnergyConsumption', labels={'x': 'Hour of Day'}),
            html.Ul([
                html.Li("Box plot shows medians and outliers by hour.", style={'fontWeight': 'bold'}),
                html.Li("Midday usage is typically higher due to business hours.", style={'fontWeight': 'bold'})
            ])
        ),

        graph_card(
            "Energy Consumption by HVAC Status",
            px.violin(df, x='HVACUsage', y='EnergyConsumption'),
            html.P("Energy is generally higher when HVAC is 'On'.")
        ),

        html.Div([
            html.Label("Select Forecast Range"),
            dcc.Dropdown(
                id='forecast-range',
                options=[
                    {'label': '1 Week', 'value': '7d'},
                    {'label': '1 Month', 'value': '30d'},
                    {'label': '1 Year', 'value': '365d'}
                ],
                value='7d',
                clearable=False,
                style={'width': '200px', 'margin': 'auto', 'marginBottom': '20px'}
            ),
        ]),

        graph_card(
            "🔮 Forecasted Energy Consumption",
            dcc.Graph(id='prophet-forecast-graph'),
            html.P("Shows forecasted consumption using Prophet model based on selected future range.")
        )

    ], style={'maxWidth': '1000px', 'margin': 'auto'}),

    html.H2("🧠 Predict Energy Consumption (Random Forest)", style={'textAlign': 'center', 'marginTop': '40px'}),

    html.Div([
        html.Div([html.Label("Temperature"), dcc.Input(id='temp', type='number', value=25, step=0.1, style={'width': '100%'})]),
        html.Div([html.Label("Humidity"), dcc.Input(id='humidity', type='number', value=50, step=0.1, style={'width': '100%'})]),
        html.Div([html.Label("Square Footage"), dcc.Input(id='area', type='number', value=1500, style={'width': '100%'})]),
        html.Div([html.Label("Occupancy"), dcc.Input(id='occupancy', type='number', value=3, style={'width': '100%'})]),
        html.Div([html.Label("Renewable Energy (kWh)"), dcc.Input(id='renewable', type='number', value=5, style={'width': '100%'})]),
        html.Div([html.Label("HVAC Usage"),
                  dcc.Dropdown(id='hvac', options=[{'label': 'On', 'value': 1}, {'label': 'Off', 'value': 0}], value=1)]),
        html.Div([html.Label("Lighting Usage"),
                  dcc.Dropdown(id='lighting', options=[{'label': 'On', 'value': 1}, {'label': 'Off', 'value': 0}], value=1)]),
        html.Button("Predict", id='predict-btn', n_clicks=0, style={'marginTop': '10px'}),
        html.Div(id='prediction-output', style={'marginTop': '20px', 'fontSize': '20px'})
    ], style={
        'maxWidth': '400px',
        'margin': 'auto',
        'display': 'flex',
        'flexDirection': 'column',
        'gap': '10px',
        'padding': '20px',
        'backgroundColor': '#f0f0f0',
        'borderRadius': '10px',
        'boxShadow': '0 2px 6px rgba(0,0,0,0.1)'
    })
])

# Callback: Manual Prediction (RandomForest)
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('temp', 'value'),
    State('humidity', 'value'),
    State('area', 'value'),
    State('occupancy', 'value'),
    State('renewable', 'value'),
    State('hvac', 'value'),
    State('lighting', 'value')
)
def predict_energy(n_clicks, temp, humidity, area, occupancy, renewable, hvac, lighting):
    if n_clicks > 0:
        input_data = pd.DataFrame([{
            'Temperature': temp,
            'Humidity': humidity,
            'SquareFootage': area,
            'Occupancy': occupancy,
            'RenewableEnergy': renewable,
            'HVACUsage': hvac,
            'LightingUsage': lighting
        }])
        prediction = model.predict(input_data)[0]
        return f"⚡ Estimated Energy Consumption: {prediction:.2f} kWh"
    return ""

# Callback: Prophet Forecast Range
@app.callback(
    Output('prophet-forecast-graph', 'figure'),
    Input('forecast-range', 'value')
)
def update_forecast_graph(range_str):
    periods = int(range_str.replace('d', ''))
    ts_data = df[['Timestamp', 'EnergyConsumption']].dropna()
    ts_data.columns = ['ds', 'y']

    model_prophet = Prophet(seasonality_mode='multiplicative')
    model_prophet.fit(ts_data)

    future = model_prophet.make_future_dataframe(periods=periods, freq='D')
    forecast = model_prophet.predict(future)
    future_forecast = forecast[forecast['ds'] > ts_data['ds'].max()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=future_forecast['ds'], y=future_forecast['yhat'],
        name='Forecast', line=dict(color='royalblue')
    ))
    fig.update_layout(
        title=f"Energy Consumption Forecast ({range_str})",
        xaxis_title='Date',
        yaxis_title='Predicted Energy Consumption (kWh)',
        template='plotly_white'
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)
