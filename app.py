import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set page configuration to wide mode with a custom title
st.set_page_config(page_title="Sales and Demand Forecasting", layout="wide")

# Function to check the login credentials
def check_login(username, password):
    return username == "Admin" and password == "123"

# Create a function to show the login page
def login():
    st.title("Login Page")
    
    # Input fields for username and password
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if check_login(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login Successful!")
            st.session_state.page = "Home"  # Store the selected page as "Home"
            st.rerun()

        else:
            st.error("Invalid username or password")

# Function to load any model from a file (Keras or pickle)
@st.cache_resource
def load_model_from_file(model_type, file_path):
    if model_type == 'keras':
        return load_model(file_path)
    elif model_type == 'pickle':
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    else:
        raise ValueError("Invalid model type")

# Function to process the sales data CSV
def process_sales_data(file_path, date_column='Date'):
    # Load and preprocess the sales data
    df = pd.read_csv(file_path)
    df[date_column] = pd.to_datetime(df[date_column])
    return df

# Function to plot time series data (Sales or Forecasts)
def plot_time_series(data, title, x_col, y_col, forecast_data=None):
    fig = px.line(data, x=x_col, y=y_col, title=title)
    if forecast_data is not None:
        fig.add_scatter(x=forecast_data['Date'], y=forecast_data['Quantity_Sold_Predicted'], 
                        mode='lines', name='Forecast', line=dict(color='red', dash='dash'))
    st.plotly_chart(fig)

# Function to handle user input for monthly demand prediction
def demand_input_form():
    month_dict = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
    }

    holiday_dict = {
        "Yes" : 1, "No" : 0
    }

    month_name = st.selectbox("Select Month", list(month_dict.keys()))
    month = month_dict[month_name]
    year = st.number_input("Enter Year", min_value=2010, max_value=2100, value=2023)
    holiday_name = st.selectbox("Select Holiday", list(holiday_dict.keys()))
    holiday = holiday_dict[holiday_name]
    price = st.number_input("Enter Price", min_value=0.0, value=30.0)
    
    return month, year, holiday, price

# Function to create forecasting sequence for time series
def create_sequences(data, target, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(target[i+sequence_length])
    return np.array(X), np.array(y)

# Function for forecasting using a deep learning model
def forecast_future(model, last_sequence, future_steps, scaler):
    future_predictions = []
    current_sequence = last_sequence  # Start with the last sequence from the dataset

    for _ in range(future_steps):
        next_prediction = model.predict(current_sequence[np.newaxis, :, :])
        future_predictions.append(next_prediction[0, 0])
        current_sequence = np.roll(current_sequence, shift=-1, axis=0)
        current_sequence[-1, -1] = next_prediction  # Update the last feature with the prediction

    # Inverse transform the predictions to the original scale
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions

# Function to display the Home Page
def home():
    # Set the title of the page
    st.title("Sales and Demand Forecasting Application")

    # Add some introductory text about the project
    st.markdown("""
    ## Overview

    Welcome to the **Sales and Demand Forecasting** application! This application is designed to help businesses predict future sales and demand based on historical data and various influencing factors. The app uses machine learning models to make forecasts and generate reports for weekly and monthly sales, helping businesses optimize inventory, pricing, and marketing strategies.

    ## Features

    - **Sales Forecasting**: 
        - Predict future sales based on historical data and key factors like price, revenue, holidays, promotions, and economic indicators.
        - The app uses deep learning models to provide accurate forecasts of product sales for the next several weeks.
    
    - **Demand Forecasting**: 
        - Predict demand for a given product based on various inputs such as month, year, holiday, and price.
        - The demand forecast is compared to actual sales data, providing insights into product demand patterns.
    
    - **Weekly Sales Report**: 
        - Generate a report showing weekly sales performance and forecast future weekly sales using SARIMAX models.

    - **Monthly Sales Report**: 
        - Generate a report showing monthly sales performance and forecast future monthly sales using SARIMAX models.

    ## How It Works

    1. **Login**: Start by logging into the app with the credentials.
    2. **Navigate**: Once logged in, navigate between different sections using the sidebar menu.
    3. **Forecasting**: In the **Sales Forecasting** and **Demand Forecasting** sections, you can input various parameters to generate predictions.
    4. **Reports**: In the **Weekly Report** and **Monthly Report** sections, you can view sales data visualizations and forecasted values.
    5. **Interactive Visualizations**: All forecasted results are displayed with interactive plots for better analysis and decision-making.

    ## Technologies Used

    - **Streamlit**: A powerful framework to create interactive web applications with Python.
    - **TensorFlow/Keras**: For deep learning-based sales forecasting.
    - **SARIMAX (Statsmodels)**: For time series forecasting.
    - **Plotly**: For creating interactive visualizations of sales data and predictions.
    - **Scikit-learn**: For data preprocessing and scaling.
    - **Pickle**: For loading pre-trained models.

    ## Getting Started

    1. Log in to the application using the credentials provided.
    2. Navigate to the desired forecasting or report section using the sidebar.
    3. Enter the necessary inputs for generating forecasts and reports.
    4. Visualize the results through interactive charts and tables.

    For any assistance or further details, feel free to contact the app administrator.

    """)

# Sales Forecasting Page
def sales_forecasting():
    st.title("Sales Forecasting")
    df = process_sales_data('sales_forecast_processed.csv')
    df = df.drop(columns=['Product_ID'])  # Drop Product_ID column

    # Resample the data to weekly frequency and calculate the mean
    weekly_df = df.resample('W', on='Date').mean().reset_index()
    features = weekly_df[['Price', 'Revenue', 'Holiday', 'Promotion', 'Economic_Indicator', 'Competitor_Price', 'Marketing_Expenditure']]
    target = weekly_df['Quantity_Sold']

    # Scale the data
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_target = scaler.fit_transform(target.values.reshape(-1, 1))

    # Create sequences for the model
    sequence_length = 12  # Use past 12 weeks to predict the next week
    X, y = create_sequences(scaled_features, scaled_target, sequence_length)

    # Load the model
    model = load_model_from_file('keras', 'sales_forcasting.h5')

    # Input field for the number of future steps
    future_steps = st.number_input("Enter the number of future weeks to forecast:", min_value=1, max_value=104, value=3)

    if st.button("Forecast"):
        # Forecast future values
        future_predictions = forecast_future(model, X[-1], future_steps, scaler)

        # Create future date range
        last_date = weekly_df['Date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=future_steps, freq='W')

        # Create a DataFrame for the future predictions
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Quantity_Sold_Predicted': future_predictions.flatten()
        })

        # Plot historical and forecasted data
        plot_time_series(weekly_df, f"Sales Forecast for the Next {future_steps} Weeks", 'Date', 'Quantity_Sold', future_df)

        # Display the predicted data
        st.subheader("Predicted Sales Data")
        st.dataframe(future_df)

# Demand Forecasting Page
def demand_forecasting():
    st.title("Demand Forecasting")
    df = process_sales_data('sales_forecast_processed.csv')

    # Monthly aggregation of sales data
    monthly_sales = df.groupby(['Product_ID', 'Year', 'Month'])[['Quantity_Sold', 'Price', 'Holiday']].sum().reset_index()

    month, year, holiday, price = demand_input_form()

    if st.button("Predict Demand"):
        # Load the demand prediction model
        rf_model = load_model_from_file('pickle', 'demand_model.pkl')

        # Prepare input data for the model
        input_data = pd.DataFrame({
            'Month': [month],
            'Year': [year],
            'Holiday': [holiday],
            'Price': [price]
        })

        # Predict demand
        predicted_demand = rf_model.predict(input_data)[0]
        st.write(f"Predicted Demand for {month}/{year}: {predicted_demand:.2f}")

        # Visualize the actual sales and predicted demand
        actual_sales = monthly_sales[monthly_sales['Year'] == year]['Quantity_Sold'].values.tolist()
        fig = go.Figure()

        # Add the actual sales data for the year
        fig.add_trace(go.Scatter(
            x=list(range(1, 13)),
            y=actual_sales,
            mode='lines+markers',
            name='Actual Sales',
            line=dict(color='blue'),
            marker=dict(symbol='circle', size=8),
        ))

        # Add the prediction for the selected month as a red marker
        fig.add_trace(go.Scatter(
            x=[month],
            y=[predicted_demand],
            mode='markers',
            name='Predicted Demand',
            marker=dict(color='red', size=10),
        ))

        # Customize the layout
        fig.update_layout(
            title="Monthly Demand Forecast vs Actual Sales",
            xaxis_title="Month",
            yaxis_title="Quantity Sold",
            showlegend=True,
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            ),
            yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='gray', zeroline=False),
        )

        # Display the plot
        st.plotly_chart(fig)

        # Plot monthly sales for each product (P001, P002, P003)
        st.subheader("Monthly Sales by Product")

        # Create a new figure for plotting monthly sales by product
        fig2 = go.Figure()

        # Add a line for each product (P001, P002, P003)
        for product in ['P001', 'P002', 'P003']:
            product_data = monthly_sales[monthly_sales['Product_ID'] == product]
            fig2.add_trace(go.Scatter(
                x=product_data['Month'],
                y=product_data['Quantity_Sold'],
                mode='lines+markers',
                name=f'Product {product}',
                line=dict(width=2),
                marker=dict(symbol='circle', size=8)
            ))
        
        # Add the prediction for the selected month as a red marker
        fig2.add_trace(go.Scatter(
            x=[month],
            y=[predicted_demand],
            mode='markers',
            name='Predicted Demand',
            marker=dict(color='red', size=10),
        ))

        # Customize the layout for the product sales plot with a larger size
        fig2.update_layout(
            title="Monthly Sales by Product",
            xaxis_title="Month",
            yaxis_title="Quantity Sold",
            showlegend=True,
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            ),
            yaxis=dict(showgrid=True, zeroline=False),
            # Increase the size of the graph
            height=800,  # height of the graph
        )

        # Display the plot with larger size
        st.plotly_chart(fig2)

def weekly_report():
    st.title("Weekly Sales Forecast Report")

    # Load the CSV file containing the sales data
    df = pd.read_csv('sales_forecast_processed.csv')
    
    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract week and year information from the Date
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Year'] = df['Date'].dt.year

    # Group the data by year and week and sum the 'Quantity_Sold'
    weekly_sales = df.groupby(['Year', 'Week'])['Quantity_Sold'].sum().reset_index()

    # Create a Date column from Year and Week for proper plotting
    weekly_sales['Date'] = pd.to_datetime(weekly_sales['Year'].astype(str) + '-W' + weekly_sales['Week'].astype(str) + '-1', format='%Y-W%U-%w')
    weekly_sales.set_index('Date', inplace=True)

    # Input field for the number of forecast steps (weeks)
    forecast_steps = st.number_input("Enter the number of forecast weeks:", min_value=1, max_value=104, value=12)

    # Button to generate the forecasted plot
    if st.button("Generate Weekly Report"):
        # Load the pre-trained SARIMA model
        @st.cache_resource
        def load_sarimax_model():
            with open('sarimax_weekly_model.pkl', 'rb') as f:
                return pickle.load(f)

        weekly_results = load_sarimax_model()

        # Forecast the future weeks using the loaded model
        forecast = weekly_results.get_forecast(steps=forecast_steps)
        forecast_index = pd.date_range(start=weekly_sales.index[-1] + pd.Timedelta(weeks=1), periods=forecast_steps, freq='W')
        forecast_values = forecast.predicted_mean
        forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecast_Quantity_Sold': forecast_values})

        # Create a Plotly figure
        fig = go.Figure()

        # Add the historical data as a line plot
        fig.add_trace(go.Scatter(
            x=weekly_sales.index,
            y=weekly_sales['Quantity_Sold'],
            mode='lines',
            name='Historical Weekly Sales',
            line=dict(color='blue')
        ))

        # Add the forecasted data as a dashed red line
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Forecast_Quantity_Sold'],
            mode='lines',
            name=f'Forecasted Weekly Sales ({forecast_steps} Weeks)',
            line=dict(color='red', dash='dash')
        ))

        # Update the layout
        fig.update_layout(
            title=f'Weekly Quantity Sold Forecast for Next {forecast_steps} Weeks',
            xaxis_title='Date',
            yaxis_title='Quantity Sold',
            template='plotly',
            xaxis=dict(
                tickformat='%Y-%m-%d',
                tickangle=45
            ),
            yaxis=dict(showgrid=True)
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)

        # Display the forecasted data in a table
        st.subheader("Forecasted Weekly Sales Data")
        st.dataframe(forecast_df)

def monthly_report():
    st.title("Monthly Sales Forecast Report")

    # Load the CSV file containing the sales data
    df = pd.read_csv('sales_forecast_processed.csv')
    
    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract month and year from the Date column
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    # Group the data by year and month and sum the 'Quantity_Sold'
    monthly_sales = df.groupby(['Year', 'Month'])['Quantity_Sold'].sum().reset_index()

    # Create a Date column from Year and Month for proper plotting
    monthly_sales['Date'] = pd.to_datetime(monthly_sales['Year'].astype(str) + '-' + monthly_sales['Month'].astype(str) + '-01', format='%Y-%m-%d')
    monthly_sales.set_index('Date', inplace=True)

    # Input field for the number of forecast steps (months)
    forecast_steps = st.number_input("Enter the number of forecast months:", min_value=1, max_value=24, value=12)

    # Button to generate the forecasted plot
    if st.button("Generate Monthly Report"):
        # Load the pre-trained SARIMAX model
        @st.cache_resource
        def load_sarimax_monthly_model():
            with open('sarimax_monthly_model.pkl', 'rb') as f:
                return pickle.load(f)

        monthly_results = load_sarimax_monthly_model()

        # Forecast the future months using the loaded model
        forecast = monthly_results.get_forecast(steps=forecast_steps)
        forecast_index = pd.date_range(start=monthly_sales.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='M')
        forecast_values = forecast.predicted_mean
        forecast_df_monthly = pd.DataFrame({'Date': forecast_index, 'Forecast_Quantity_Sold': forecast_values})

        # Create a Plotly figure
        fig = go.Figure()

        # Add the historical data as a line plot
        fig.add_trace(go.Scatter(
            x=monthly_sales.index,
            y=monthly_sales['Quantity_Sold'],
            mode='lines',
            name='Historical Monthly Sales',
            line=dict(color='green')
        ))

        # Add the forecasted data as a dashed orange line
        fig.add_trace(go.Scatter(
            x=forecast_df_monthly['Date'],
            y=forecast_df_monthly['Forecast_Quantity_Sold'],
            mode='lines',
            name=f'Forecasted Monthly Sales ({forecast_steps} Months)',
            line=dict(color='orange', dash='dash')
        ))

        # Update the layout
        fig.update_layout(
            title=f'Monthly Quantity Sold Forecast for Next {forecast_steps} Months',
            xaxis_title='Date',
            yaxis_title='Quantity Sold',
            template='plotly',
            xaxis=dict(
                tickformat='%Y-%m-%d',
                tickangle=45
            ),
            yaxis=dict(showgrid=True)
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)

        # Display the forecasted data in a table
        st.subheader("Forecasted Monthly Sales Data")
        st.dataframe(forecast_df_monthly)

# Sidebar navigation to switch between pages
def sidebar_navigation():
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        # Show login screen if not logged in
        st.sidebar.title("Navigation")
        options = st.sidebar.radio("Go to", ["Login"])
        
        if options == "Login":
            login()
    else:
        # Navigation for logged-in users
        st.sidebar.title("Navigation")
        options = st.sidebar.radio("Go to", ["Home", "Sales Forecasting", "Demand Forecasting", "Weekly Report", "Monthly Report"])
        
        if options == "Home":
            home()
        elif options == "Sales Forecasting":
            sales_forecasting()
        elif options == "Demand Forecasting":
            demand_forecasting()
        elif options == "Weekly Report":
            weekly_report()
        elif options == "Monthly Report":
            monthly_report()

        # Logout button as a separate button
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            del st.session_state.username
            st.session_state.page = "Login"
            st.rerun()

# Check if the user is already logged in
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    login()
else:
    sidebar_navigation()
