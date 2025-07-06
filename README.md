# ⚡ Energy Consumption Dashboard

An interactive web dashboard for analyzing, predicting, and forecasting energy usage based on real-world features like temperature, humidity, occupancy, HVAC, lighting, and renewable energy usage.

Built with **Dash**, **Plotly**, **Scikit-learn**, and **Facebook Prophet**.

---

## 🚀 Features

* 📊 **Energy Visualizations**:

  * Time-series energy usage trends
  * Hourly usage distributions
  * Feature correlation heatmap
  * HVAC impact on consumption

* 🔮 **Forecasting** (Prophet):

  * Select from **1 Week**, **1 Month**, or **1 Year** future forecasts
  * Displays clean forecast line using Prophet (no clutter or confidence bands)

* 🧠 **Manual Prediction** (Random Forest):

  * Input values like temperature, occupancy, etc.
  * Predict energy usage instantly using a trained ML model

* 🎛️ **Modern Interface**:

  * Responsive cards, styled with shadows and rounded corners
  * Intuitive dropdowns and labeled input fields
  * Modular code structure for easy updates

---

## 📊 Models Used

| Model                    | Purpose                     | Input                                                    |
| ------------------------ | --------------------------- | -------------------------------------------------------- |
| 🧠 RandomForestRegressor | Predict energy from inputs  | Temperature, Humidity, SquareFootage, Occupancy, HVAC... |
| 📈 Prophet (Meta)        | Forecast future consumption | Timestamp and historical EnergyConsumption               |

---

## 📁 File Structure

```
energy-dashboard/
├── Energy_consumption.csv         # Input dataset
├── app.py                         # Main Dash application
├── README.md                      # You're reading this!
├── requirements.txt               # Python dependencies
```

---

## 💡 Example Use Cases

* Determine how temperature and occupancy affect energy
* Forecast next month's energy consumption
* Estimate energy usage for a new building design

---

## 📦 Installation

1. Clone the repo:
   git clone [https://github.com/your-username/energy-dashboard.git](https://github.com/your-username/energy-dashboard.git)
   cd energy-dashboard

2. Install dependencies:
   pip install -r requirements.txt

3. Run the app:
   python app.py

4. Open your browser and visit:
   [http://127.0.0.1:8050](http://127.0.0.1:8050)

---

## 👨‍💻 Author

**Muhammad Bakhtiar Bin Yusof**
GitHub: [https://github.com/MYbakhtiardev](https://github.com/MYbakhtiardev)
LinkedIn: [https://linkedin.com/in/muhammad-bakhtiar-47047827a](https://linkedin.com/in/muhammad-bakhtiar-47047827a)

---

## Data Source
https://www.kaggle.com/datasets/mrsimple07/energy-consumption-prediction

## 📄 License

This project is licensed under the MIT License.

---

