import pandas as pd
import yfinance as yf
import json
import streamlit as st
from plotly import graph_objs as go
import plotly.express as px

st.set_page_config(layout="wide")

# Sidebar Settings
st.sidebar.title("Stock Forecast Settings")
st.sidebar.subheader("Select Dataset")
ticker_list = ( 'AZN', 'HMC',"CSCO","PG")
ticker = st.sidebar.selectbox('Choose Stock:', ticker_list)

st.sidebar.subheader("Visualization Options")
show_predicted = st.sidebar.checkbox("Show Predicted Line", value=True)
show_actual = st.sidebar.checkbox("Show Actual Line", value=True)
show_test = st.sidebar.checkbox("Show Test Line", value=True)

# Define file paths dynamically
news_title_path = f"../news_title/{ticker}_news_title.json"
sentiment_score_path = f"../sentiment_score/{ticker}_news_sentiment_score.json"
hybrid_predictions_path = f"../predictions/{ticker}_hybrid_predictions.csv"
model_evaluation_result_path = f"../model_evaluation/{ticker}_model_evaluation.json"
# Load hybrid predictions
hybrid_predictions = pd.read_csv(hybrid_predictions_path)

# Load sentiment score data
with open(sentiment_score_path) as f:
    sentiment_data = json.load(f)

# Convert sentiment data to DataFrame
sentiment_df = pd.DataFrame(sentiment_data).transpose()
sentiment_df.columns = [f"Score_{i+1}" for i in range(sentiment_df.shape[1])]
sentiment_df.index.name = "Date"
sentiment_df.reset_index(inplace=True)

# Load news titles data
news_df = pd.read_json(news_title_path, lines=True)

# Convert 'Date' in news_df to match sentiment_df's format
news_df['Date'] = pd.to_datetime(news_df['Date'], unit='ms').dt.strftime('%Y-%m-%d')

# Merge the two DataFrames on the 'Date' column
merged_df = pd.merge(news_df, sentiment_df, on="Date", how="inner")
merged_df["Date"] = pd.to_datetime(merged_df["Date"])

st.title('Stock Forecast App')

# Define date range
START = merged_df["Date"].min().strftime('%Y-%m-%d')
END = merged_df["Date"].max().strftime('%Y-%m-%d')

@st.cache_data
def load_data(ticker):
    historical_data = yf.Ticker(ticker).history(start=START, end=END)

    return historical_data

data = load_data(ticker)


stock = yf.Ticker(ticker)

company_name = stock.info.get('longName', 'Name not available')
stock_display_name = f"{ticker} - {company_name}"

# Display summary in columns
col1, col2, col3 = st.columns(3)
col1.metric("Selected Stock", stock_display_name)
col2.metric("Data Start Date", START)
col3.metric("Data End Date", END)

# Display News and Sentiment Data
st.subheader("News Data (First and Last Rows)")
with st.expander("View Merged Data"):
    st.table(merged_df)



# Trim data from the start date of hybrid_predictions
hybrid_start_date = hybrid_predictions['Date'].iloc[0]
train_data = data[data.index < hybrid_start_date]

# Create Interactive Chart
fig = go.Figure()

# Add actual stock prices
fig.add_trace(go.Scatter(
    x=train_data.index, 
    y=train_data['Close'],
    marker=dict(color="red"),  # Red for actual values
    mode='lines', 
    name="Actual Stock Close",
    visible=show_actual  # Control visibility via checkbox
))

# Add predicted stock prices
fig.add_trace(go.Scatter(
    x=pd.to_datetime(hybrid_predictions['Date']), 
    y=hybrid_predictions['Predicted'],
    marker=dict(color="blue"),  # Blue for predicted values
    mode='lines', 
    name="Predicted Stock Close",
    visible=show_predicted  # Control visibility via checkbox
))

# Add test stock prices
actual_test = data.tail(len(hybrid_predictions))
fig.add_trace(go.Scatter(
    x=actual_test.index, 
    y=actual_test['Close'],
    marker=dict(color="green"),  # Green for test data values
    mode='lines', 
    name="Test Stock Close",
    visible=show_test  # Control visibility via checkbox
))

# Update layout with toggles
fig.layout.update(
    title_text='Time Series Data with Rangeslider',
    xaxis_rangeslider_visible=True
)
# Display the chart
st.subheader("Interactive Stock Prediction Chart")
st.plotly_chart(fig)

# Chart Updates
fig.update_layout(
    title="Stock Prediction and Analysis",
    xaxis_title="Date",
    yaxis_title="Stock Price",
    legend_title="Legend",
    template="plotly_white"
)


max_date_minus_one = data.index.max() - pd.Timedelta(days=-1)

# ROI Calculation Section (Fix nearest date logic)
st.sidebar.subheader("Investment Simulator")
investment_date = st.sidebar.date_input(
    "Investment Date", 
    value=pd.to_datetime(START),  
    min_value=pd.to_datetime(data.index.min()),  
    max_value=max_date_minus_one
)

end_date = st.sidebar.date_input(
    "End Date", 
    value=pd.to_datetime(END) ,  
    min_value=pd.to_datetime(data.index.min()),  
    max_value=max_date_minus_one
)
investment_amount = st.sidebar.number_input("Investment Amount ($)", value=1000.0, min_value=1.0)

# Add a dropdown to select data source
forecast_source = st.sidebar.selectbox(
    "Forecast Source",
    options=["Hybrid Predictions", "Actual Data"],
    index=0  # Default to the first option
)

# Ensure the data index is timezone-naive
data.index = pd.to_datetime(data.index).tz_localize(None)

# Convert input dates to timezone-naive datetime
investment_date = pd.to_datetime(investment_date).tz_localize(None)
end_date = pd.to_datetime(end_date).tz_localize(None)


# Get the start date for hybrid predictions
hybrid_start_date = pd.to_datetime(hybrid_predictions['Date'].iloc[0])  # First date of hybrid predictions

# Filter pre-prediction data (all data before the start of hybrid predictions)
pre_prediction_data = data[data.index < hybrid_start_date]

# Ensure 'Date' in hybrid_predictions is in datetime format
hybrid_predictions['Date'] = pd.to_datetime(hybrid_predictions['Date'])

# Prepare hybrid predictions: Replace 'Close' values with 'Predicted' starting from hybrid_start_date
pre_prediction_data_reset = pre_prediction_data[['Close']].reset_index()

# Merge the pre-prediction data with hybrid predictions
hybrid_predictions = hybrid_predictions[['Date', 'Predicted']]  # Only keep 'Date' and 'Predicted'

# Create the final combined data, where we replace the 'Close' column after the last actual data point
combined_data = pd.concat([pre_prediction_data_reset, hybrid_predictions.rename(columns={'Predicted': 'Close'})], axis=0)
combined_data["Date"] = pd.to_datetime(combined_data["Date"])

# Now combined_data has only 'Date' and 'Close', with 'Close' including both actual and predicted prices
combined_data = combined_data[['Date', 'Close']]

# Reset the index to ensure 'Date' is a column, not an index
combined_data.set_index('Date', inplace=True)
# Check the first few rows of combined_data to see how the 'Date' column is formatted


# Validate date range
if investment_date < data.index.min() or end_date > data.index.max():
    st.sidebar.error("Selected dates are out of range. Please choose valid dates.")
elif investment_date >= end_date:
    st.sidebar.error("End date must be after the investment date.")
else:
    try:
        # Select the appropriate data based on the forecast source
        if forecast_source == "Actual Data":
            selected_data = data
        elif forecast_source == "Hybrid Predictions":
            selected_data = combined_data

                    
        start_idx = selected_data.index.get_loc(investment_date)
        end_idx = selected_data.index.get_loc(end_date)

        start_price =  selected_data.iloc[start_idx]["Close"]
        end_price = selected_data.iloc[end_idx]["Close"] 

        # Calculate ROI and Final Value
        roi = ((end_price - start_price) / start_price) * 100
        final_value = investment_amount * (end_price / start_price)

        # Display the results
        st.sidebar.success(f"From {data.index[start_idx].strftime('%Y-%m-%d')} to {data.index[end_idx].strftime('%Y-%m-%d')}:")
        st.sidebar.metric("ROI (%)", f"{roi:.2f}%")
        st.sidebar.metric("Final Investment Value ($)", f"{final_value:.2f}")
    except Exception as e:
        st.sidebar.error(f"An error occurred: {e}")


col1,col2,col3 = st.columns(3)

# Resetting the index for both datasets to align them by position
hybrid_predictions = hybrid_predictions.reset_index(drop=True)

# Extract 'Close' values as a NumPy array from actual_test DataFrame
test = actual_test['Close'].values

# Convert hybrid_predictions into a DataFrame with column name "Predicted"
hybrid_predictions = pd.DataFrame(hybrid_predictions, columns=["Predicted"])

# Create a DataFrame for the actual stock prices ('test' contains 'Close' values)
actual_test_df = pd.DataFrame(test, columns=["Actual_Close"])

# Concatenate the two DataFrames along columns (axis=1)
compare_df = pd.concat([hybrid_predictions, actual_test_df], axis=1)

# Check if the file exists before reading it
try:
    with open(model_evaluation_result_path, "r") as f:
        results_dict = json.load(f)
    # Convert the results_dict to a DataFrame for display in a table format
    # Extract the metrics for both models
    results_data = {
        "Metric": ["MSE", "RMSE", "MAE", "R-squared (R²)", "Testing Accuracy (%)"],

        "LSTM-ARIMA Hybrid Model": [
            results_dict["LSTM-ARIMA Hybrid Model"]["MSE"],
            results_dict["LSTM-ARIMA Hybrid Model"]["RMSE"],
            results_dict["LSTM-ARIMA Hybrid Model"]["MAE"],
            results_dict["LSTM-ARIMA Hybrid Model"]["R-squared (R²)"],
            results_dict["LSTM-ARIMA Hybrid Model"]["Testing Accuracy (%)"]
        ]
    }

    # Create a DataFrame from the dictionary
    results_df = pd.DataFrame(results_data)

    with col1:
        # Display the contents as a table
        st.write(f"Model Evaluation Results for {ticker}:")
        st.dataframe(results_df)  # Display as an interactive table

except FileNotFoundError:
    st.error(f"The file {model_evaluation_result_path} does not exist.")
    
with col2:
    st.subheader('Comparison')
    st.write(compare_df)

with col3:
    st.subheader("Summary Insights")
    st.write(f"**Selected Stock:** {ticker}")
    st.write(f"Number of predictions: {len(hybrid_predictions)}")

    st.write(f"**Prediction Date Range:** {hybrid_start_date} to {data.index.max().strftime('%Y-%m-%d')}")
    st.write(f"**Historical Data Range:** {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
    
    
