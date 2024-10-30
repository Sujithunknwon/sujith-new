import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from io import BytesIO  # For image conversion
import base64  # For image encoding

from flask import Flask, render_template, request, redirect, url_for

app = Flask(_name_)

# Load your pre-trained model (replace with your model loading logic)
model = GradientBoostingRegressor()  # Placeholder for your trained model

# Function to preprocess data (replace with your actual logic)
def preprocess_data(data):
    df = pd.DataFrame(data)  # Assuming data is a list of dictionaries
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    # ... your preprocessing steps ...
    return df

# Function to make predictions (replace with your actual logic)
def predict_deviation(df, n_parts_per_day):
    X = df[['Rolling_Mean', 'Month', 'Day', 'DayOfWeek', 'Lag_1', 'Lag_2', 'Squareness']]
    y = df['Values']
    # ... your prediction steps ...
    return predicted_values, rmse

# Function to generate plot image (replace with your actual logic)
def plot_graph(y_pred):
    fig, ax = plt.subplots(figsize=(6, 4))
    # ... your plotting steps ...

    # Convert plot to a byte array for efficient transfer
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png')
    img_bytes = img_buffer.getvalue()

    return base64.b64encode(img_bytes).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['data_file']
        n_parts = int(request.form['n_parts'])

        if uploaded_file and n_parts > 0:
            df = pd.read_excel(uploaded_file)
            df = preprocess_data(df.to_dict(orient='records'))

            predicted_values, rmse = predict_deviation(df.copy(), n_parts)
            plot_image = plot_graph(predicted_values)

            return render_template('results.html', data=df.to_dict(orient='records'),
                                   predicted_values=predicted_values.tolist(),
                                   rmse=rmse, plot_image=plot_image)

        return redirect(url_for('index'), error="Invalid file or number of parts")

    return render_template('index.html')

if _name_ == '_main_':
    app.run(debug=True)  # Set debug=False for production
