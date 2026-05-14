import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Supply Chain Analytics Platform", layout="wide")

st.title("🚀 Supply Chain Analytics Platform")

tab1, tab2, tab3 = st.tabs(["📊 Demand Histogram", "📈 Demand Forecasting", "📦 Inventory Optimization"])

with tab1:
    st.header("Demand Histogram Analyzer")
    
    # --- Data Configuration ---
    st.subheader("1. Data Configuration")
    data_source = st.radio("Select Data Source:", ("Generate Synthetic Data", "Upload Your Own Data"), horizontal=True, key="ds_p1")

    df = None

    if data_source == "Generate Synthetic Data":
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            dist_type = st.selectbox("Distribution Type", ("Normal", "Poisson", "Uniform"), key="dist_p1")
        with col_b:
            avg_demand = st.number_input("Average Demand", min_value=1.0, value=100.0, key="avg_p1")
        with col_c:
            num_periods = st.number_input("Number of Periods", min_value=10, value=10000, key="periods_p1")
        with col_d:
            if dist_type == "Normal":
                variation = st.number_input("Std Dev (Variation)", min_value=0.1, value=15.0, key="v_norm")
            elif dist_type == "Uniform":
                variation = st.number_input("Range (+/-)", min_value=1.0, value=30.0, key="v_uni")
            else:
                st.markdown("<p style='padding-top:25px; color:gray;'>Poisson variation fixed by Mean.</p>", unsafe_allow_html=True)

        np.random.seed(42)
        if dist_type == "Normal":
            generated = np.random.normal(avg_demand, variation, num_periods)
        elif dist_type == "Poisson":
            generated = np.random.poisson(avg_demand, num_periods)
        else:
            # FIX: Use floor to ensure we don't round UP above the max limit
            generated = np.random.uniform(avg_demand - variation, avg_demand + variation, num_periods)
        
        df = pd.DataFrame({'Demand': np.floor(np.clip(generated, 0, None))})

    # --- Analysis & Visualization ---
    if df is not None:
        st.divider()
        
        # New Feature: Threshold Analysis
        st.subheader("2. Threshold & Service Level Analysis")
        t_col1, t_col2 = st.columns([1, 2])
        with t_col1:
            threshold = st.number_input("Check points below value:", value=40.0, step=1.0)
            count_below = len(df[df['Demand'] < threshold])
            percent_below = (count_below / len(df)) * 100
            st.metric("Points Below Threshold", f"{count_below}", f"{percent_below:.1f}% of total")
        with t_col2:
            st.info(f"💡 This indicates that in **{percent_below:.1f}%** of periods, demand was lower than {threshold}. In inventory terms, this helps you visualize your service level or potential stock-out risk.")

        st.divider()
        st.subheader("3. Visual Distribution")
        num_bins = st.slider("Select Number of Bins:", 5, 50, 15)
        
        # Main Display: Chart, Table, and Summary
        chart_col, table_col = st.columns([2, 1])

        with chart_col:
            fig = px.histogram(df, x="Demand", nbins=num_bins, template="plotly_white", color_discrete_sequence=['#4F8BF9'])
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
            
            # Data Summary Table (Requested)
            st.markdown("#### 📋 Statistical Summary")
            summary_stats = df['Demand'].describe().to_frame().T
            st.dataframe(summary_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']], use_container_width=True)

        with table_col:
            st.markdown("#### Bin Frequency Table")
            counts, bin_edges = np.histogram(df['Demand'], bins=num_bins)
            bin_df = pd.DataFrame({
                "Bin Range": [f"{int(bin_edges[i])} - {int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)],
                "Count": counts,
                "%": (counts / len(df) * 100).round(1)
            })
            st.dataframe(bin_df, use_container_width=True, hide_index=True)
