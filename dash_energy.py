import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
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

# Prophet Forecast
ts_data = df[['Timestamp', 'EnergyConsumption']].dropna()
ts_data.columns = ['ds', 'y']
model_prophet = Prophet(seasonality_mode='multiplicative')
model_prophet.fit(ts_data)
future = model_prophet.make_future_dataframe(periods=24, freq='h')
forecast = model_prophet.predict(future)
fig_forecast = go.Figure()

fig_forecast.add_trace(go.Scatter(
    x=forecast['ds'], y=forecast['yhat'],
    name='Forecast', line=dict(color='royalblue')
))
fig_forecast.add_trace(go.Scatter(
    x=forecast['ds'], y=forecast['yhat_upper'],
    name='Upper Bound', line=dict(dash='dot', color='lightblue')
))
fig_forecast.add_trace(go.Scatter(
    x=forecast['ds'], y=forecast['yhat_lower'],
    name='Lower Bound', line=dict(dash='dot', color='lightblue'),
    fill='tonexty', fillcolor='rgba(173, 216, 230, 0.2)'
))

fig_forecast.update_layout(
    title='Forecasted Energy Consumption (Next 24 Hours)',
    xaxis_title='Timestamp',
    yaxis_title='Energy Consumption (kWh)',
    template='plotly_white'
)

# Dash app
app = dash.Dash(__name__)

def graph_card(title, figure, explanation):
    return html.Div([
        dcc.Graph(figure=figure),
        html.P(explanation, style={'padding': '10px', 'fontStyle': 'italic'})
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
            "This line graph shows how energy consumption varies over time. Peaks often correlate with high occupancy or HVAC usage."
        ),

        graph_card(
            "Feature Correlation Heatmap",
            px.imshow(df.select_dtypes(include=['float64', 'int64']).corr()),
            "This heatmap highlights relationships between features. For example, square footage and HVAC usage may strongly influence energy."
        ),

        graph_card(
            "Hourly Energy Consumption Distribution",
            px.box(df, x=df['Timestamp'].dt.hour, y='EnergyConsumption', labels={'x': 'Hour of Day'}),
            "This box plot shows energy trends throughout the day. Peaks during working hours suggest behavior-linked usage."
        ),

        graph_card(
            "Energy Consumption by HVAC Status",
            px.violin(df, x='HVACUsage', y='EnergyConsumption'),
            "The violin plot shows energy distribution based on HVAC usage. Energy is generally higher when HVAC is 'On'."
        ),

        graph_card(
            "🔮 Forecasted Energy Consumption (Next 24h)",
            fig_forecast,
            "Forecast using Prophet model. This shows expected consumption trends in the upcoming 24 hours based on historical data."
        ),
    ], style={'maxWidth': '1000px', 'margin': 'auto'}),

    html.H2("🧠 Predict Energy Consumption", style={'textAlign': 'center', 'marginTop': '40px'}),

    html.Div([
        html.Div([
            html.Label("Temperature"),
            dcc.Input(id='temp', type='number', value=25, step=0.1, style={'width': '100%'})
        ], style={'marginBottom': '15px'}),

        html.Div([
            html.Label("Humidity"),
            dcc.Input(id='humidity', type='number', value=50, step=0.1, style={'width': '100%'})
        ], style={'marginBottom': '15px'}),

        html.Div([
            html.Label("Square Footage"),
            dcc.Input(id='area', type='number', value=1500, style={'width': '100%'})
        ], style={'marginBottom': '15px'}),

        html.Div([
            html.Label("Occupancy"),
            dcc.Input(id='occupancy', type='number', value=3, style={'width': '100%'})
        ], style={'marginBottom': '15px'}),

        html.Div([
            html.Label("Renewable Energy (kWh)"),
            dcc.Input(id='renewable', type='number', value=5, style={'width': '100%'})
        ], style={'marginBottom': '15px'}),

        html.Div([
            html.Label("HVAC Usage"),
            dcc.Dropdown(
                id='hvac',
                options=[{'label': 'On', 'value': 1}, {'label': 'Off', 'value': 0}],
                value=1,
                style={'width': '100%'}
            )
        ], style={'marginBottom': '15px'}),

        html.Div([
            html.Label("Lighting Usage"),
            dcc.Dropdown(
                id='lighting',
                options=[{'label': 'On', 'value': 1}, {'label': 'Off', 'value': 0}],
                value=1,
                style={'width': '100%'}
            )
        ], style={'marginBottom': '15px'}),

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

if __name__ == '__main__':
    app.run(debug=True)
