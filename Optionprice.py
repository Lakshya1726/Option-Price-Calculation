# Streamlit app for Black-Scholes Option Pricing
import streamlit as st
import numpy as np
from scipy.stats import norm


# Black-Scholes Pricing Function
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Black-Scholes price of a European option.

    :param S: Current stock price
    :param K: Strike price
    :param T: Time to maturity (in years)
    :param r: Risk-free rate (annualized)
    :param sigma: Volatility (annualized)
    :param option_type: 'call' or 'put'
    :return: Option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return price


# Streamlit App Layout
st.title("Black-Scholes Option Pricing Model")
st.write("This app calculates the price of a European option using the Black-Scholes formula.")

# Input Section
st.sidebar.header("Option Parameters")
S = st.sidebar.number_input("Current Stock Price (S)", min_value=0.01, value=100.0, step=0.01)
K = st.sidebar.number_input("Strike Price (K)", min_value=0.01, value=100.0, step=0.01)
T = st.sidebar.number_input("Time to Maturity (T, in years)", min_value=0.01, value=1.0, step=0.01)
r = st.sidebar.number_input("Risk-Free Interest Rate (r, as decimal)", min_value=0.0, value=0.05, step=0.01)
sigma = st.sidebar.number_input("Volatility (sigma, as decimal)", min_value=0.01, value=0.2, step=0.01)
option_type = st.sidebar.selectbox("Option Type", ["call", "put"])

# Calculate Price
if st.sidebar.button("Calculate Option Price"):
    price = black_scholes(S, K, T, r, sigma, option_type)
    st.write(f"The {option_type} option price is: **${price:.2f}**")

# File Upload Section
st.header("Batch Processing")
st.write("Upload a CSV file with columns: `S`, `K`, `T`, `r`, `sigma`, `option_type`.")
uploaded_file = st.file_uploader("Upload your file", type=["csv"])

if uploaded_file:
    import pandas as pd

    # Read uploaded CSV
    data = pd.read_csv(uploaded_file)

    # Validate required columns
    required_columns = {"S", "K", "T", "r", "sigma", "option_type"}
    if not required_columns.issubset(data.columns):
        st.error(f"CSV file must contain columns: {', '.join(required_columns)}")
    else:
        # Calculate option prices
        data['option_price'] = data.apply(
            lambda row: black_scholes(
                row['S'], row['K'], row['T'], row['r'], row['sigma'], row['option_type']
            ), axis=1
        )
        st.write("Option prices calculated for the uploaded data:")
        st.dataframe(data)

        # Allow file download
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="option_pricing_results.csv",
            mime="text/csv",
        )
