import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

# Set page configuration
st.set_page_config(page_title="Supply Chain Analytics Platform", layout="wide")

st.title("🚀 Supply Chain Analytics Platform")

# --- CREATE TABS FOR PAGES ---
tab1, tab2, tab3 = st.tabs(["📊 Demand Histogram", "📈 Demand Forecasting", "📦 Inventory Optimization"])

# ==========================================
# PAGE 1: DEMAND HISTOGRAM
# ==========================================
with tab1:
    st.header("Demand Histogram Analyzer")
    st.markdown("Analyze historical demand patterns or simulate theoretical distributions to understand demand variability.")
    
    # --- Data Configuration Section ---
    st.subheader("1. Data Configuration")
    data_source = st.radio("Select Data Source:", ("Generate Synthetic Data", "Upload Your Own Data"), horizontal=True, key="data_src_p1")

    df = None

    if data_source == "Generate Synthetic Data":
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            dist_type = st.selectbox("Distribution Type", ("Normal", "Poisson", "Uniform"), key="dist_p1")
        with col_b:
            avg_demand = st.number_input("Average Demand", min_value=1.0, value=100.0, key="avg_p1")
        with col_c:
            num_periods = st.number_input("Number of Periods", min_value=10, max_value=10000, value=365, key="periods_p1")
        with col_d:
            if dist_type == "Normal":
                variation = st.number_input("Std Dev (Variation)", min_value=0.1, value=15.0, key="var_normal_p1")
            elif dist_type == "Uniform":
                variation = st.number_input("Range (+/-)", min_value=1.0, value=30.0, key="var_uni_p1")
            else:
                st.markdown("<p style='padding-top:25px; color:gray;'>Poisson variation is tied to the Mean.</p>", unsafe_allow_html=True)

        # Data Generation
        np.random.seed(42)
        if dist_type == "Normal":
            generated = np.random.normal(avg_demand, variation, num_periods)
        elif dist_type == "Poisson":
            generated = np.random.poisson(avg_demand, num_periods)
        else:
            generated = np.random.uniform(avg_demand - variation, avg_demand + variation, num_periods)
        
        df = pd.DataFrame({'Demand': np.round(np.clip(generated, 0, None), 0)})

    else:
        # Upload Logic
        col_up1, col_up2 = st.columns([1, 2])
        with col_up1:
            st.info("Ensure your Excel file has a column named exactly 'Demand'.")
            uploaded_file = st.file_uploader("Upload .xlsx file", type=["xlsx"], key="uploader_p1")
        
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
                if 'Demand' not in df.columns:
                    st.error("❌ Column 'Demand' not found in file.")
                    df = None
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Awaiting file upload...")

    # --- Analysis & Visualization Section ---
    if df is not None:
        st.divider()
        st.subheader("2. Histogram Analysis")
        
        # Bin Slider
        num_bins = st.slider("Select Number of Bins:", min_value=5, max_value=50, value=15, key="bins_p1")
        
        chart_col, table_col = st.columns([2, 1])

        with chart_col:
            fig = px.histogram(
                df, x="Demand", nbins=num_bins, 
                title="Demand Frequency Distribution",
                template="plotly_white",
                color_discrete_sequence=['#4F8BF9']
            )
            fig.update_layout(bargap=0.1, yaxis_title="Count of Periods", xaxis_title="Demand Quantity")
            st.plotly_chart(fig, use_container_width=True)

        with table_col:
            st.markdown("#### Bin Frequency Table")
            
            # Calculate Bin Table
            counts, bin_edges = np.histogram(df['Demand'], bins=num_bins)
            
            bin_df = pd.DataFrame({
                "Bin Range": [f"{int(bin_edges[i])} - {int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)],
                "Frequency (Count)": counts
            })
            
            bin_df["% of Total"] = (bin_df["Frequency (Count)"] / len(df) * 100).round(1).astype(str) + '%'
            
            st.dataframe(bin_df, use_container_width=True, hide_index=True)

        # Summary Statistics
        with st.expander("Show Statistical Summary"):
            st.write(df.describe().T)


# ==========================================
# PAGE 2: PLACEHOLDER FOR FORECASTING
# ==========================================
with tab2:
    st.header("📈 Demand Forecasting")
    st.info("This section is reserved for future predictive analytics tools (e.g., Moving Average, Exponential Smoothing, ARIMA).")
    # You can start building your forecasting models here!


# ==========================================
# PAGE 3: PLACEHOLDER FOR INVENTORY
# ==========================================
with tab3:
    st.header("📦 Inventory Optimization")
    st.info("This section is reserved for setting Safety Stock, Reorder Points (ROP), and Economic Order Quantity (EOQ) calculations.")
    # You can start building your inventory logic here!
