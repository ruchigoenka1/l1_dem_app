import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Inventory Analysis Dashboard")

uploaded_file = st.file_uploader("Upload Inventory File", type=["xlsx", "csv"])

if uploaded_file is not None:

    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(df)

    required_columns = [
        "Date",
        "Opening Balance",
        "Demand",
        "Shipment Received",
        "Closing Balance"
    ]

    # Validate columns
    if all(col in df.columns for col in required_columns):

        # Convert date
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        st.success("File validated successfully")

        # Demand Histogram
        st.subheader("Demand Distribution")

        fig1, ax1 = plt.subplots()
        ax1.hist(df["Demand"], bins=20)
        ax1.set_xlabel("Demand")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Demand Histogram")

        st.pyplot(fig1)

        # Closing Balance Plot
        st.subheader("Daily Closing Inventory")

        fig2, ax2 = plt.subplots()
        ax2.plot(df["Date"], df["Closing Balance"], marker="o")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Closing Balance")
        ax2.set_title("Inventory Closing Balance Over Time")

        plt.xticks(rotation=45)

        st.pyplot(fig2)

    else:
        st.error("Uploaded file does not contain required columns")