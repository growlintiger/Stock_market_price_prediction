# Stock Prediction Web Application Using LSTM and Streamlit

Welcome to the Stock Prediction Web Application! This project leverages Long Short-Term Memory (LSTM) neural networks to predict stock prices using historical data. Built using Python, TensorFlow, and Streamlit, the app provides users with an intuitive interface for real-time stock analysis.

## Features
- Predict the next day’s stock price using LSTM models.
- Display real-time stock data using the Yahoo Finance API.
- View interactive visualizations with Plotly.
- Evaluate model accuracy using Mean Squared Error (MSE).
- Monitor price trends through a dynamic ticker bar.

## Live Demo
Experience the application live at:
[Stock Prediction App](https://stock-prediction-alex.streamlit.app)

## Installation Guide
Follow these steps to run the application locally.

### Prerequisites
Ensure you have the following installed:
- Python 3.9 or above
- pip (Python Package Installer)
- Virtual Environment (Optional)

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-repo/stock-prediction-app.git
cd stock-prediction-app
```

### Step 2: Create and Activate a Virtual Environment (Optional)
```bash
python -m venv env
# On Windows
env\Scripts\activate
# On Mac/Linux
source env/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

## Usage
- Visit `http://localhost:8501` in your browser.
- Enter or select a stock ticker symbol.
- View real-time stock prices and predicted prices.
- Analyze interactive graphs and model performance.

## Troubleshooting
- **ModuleNotFoundError:** Ensure dependencies are installed using `pip install -r requirements.txt`.
- **Streamlit Not Found:** Install Streamlit manually:
  ```bash
  pip install streamlit
  ```
- **TensorFlow Issues:** Ensure TensorFlow is installed correctly:
  ```bash
  pip install tensorflow
  ```

## Future Enhancements
- Implement sentiment analysis for better predictions.
- Add support for additional financial indicators.
- Enhance model accuracy using ensemble techniques.

## License
This project is licensed under the MIT License. Feel free to customize and improve it.

## Contact
For any inquiries, feel free to contact [Alex Binu](mailto:alexbinu2004@gmail.com).

---
Enjoy predicting stock prices with the Stock Prediction Web App! 🚀

