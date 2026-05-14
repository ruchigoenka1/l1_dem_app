import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

# Set page configuration
st.set_page_config(page_title="Demand Histogram Analyzer", layout="wide")

st.title("📊 Demand Histogram Analyzer")
st.markdown("""
This app helps you visualize and analyze demand patterns. You can either **generate synthetic data** based on statistical distributions or **upload your own historical demand data**.
""")

# --- SIDEBAR: DATA INPUT SELECTION ---
st.sidebar.header("1. Choose Data Source")
data_source = st.sidebar.radio("Select Data Source:", ("Generate Synthetic Data", "Upload Your Own Data"))

df = None

if data_source == "Generate Synthetic Data":
    st.sidebar.subheader("Distribution Parameters")
    
    # Distribution Type
    dist_type = st.sidebar.selectbox(
        "Distribution Type:", 
        ("Normal", "Poisson", "Uniform"),
        index=0,
        help="Normal is standard. Poisson is great for low-volume/slow-moving items. Uniform assumes equal chance."
    )
    
    # Distribution Quick Link Guide
    st.sidebar.markdown("[📊 View Distribution Shapes Guide](https://en.wikipedia.org/wiki/Probability_distribution)")
    
    # Common inputs
    avg_demand = st.sidebar.number_input("Average Demand (Mean)", min_value=1.0, value=100.0, step=5.0)
    num_periods = st.sidebar.number_input("Number of Periods (e.g., Days/Weeks)", min_value=10, max_value=10000, value=365, step=50)
    
    # Generate data based on selection
    np.random.seed(42) # For reproducibility
    
    if dist_type == "Normal":
        variation = st.sidebar.number_input("Variation (Standard Deviation)", min_value=0.1, value=15.0, step=1.0)
        generated_demand = np.random.normal(loc=avg_demand, scale=variation, size=num_periods)
        # Demand can't be negative in real life
        generated_demand = np.clip(generated_demand, 0, None) 
        
    elif dist_type == "Poisson":
        st.sidebar.info("💡 Poisson variation is tied directly to the mean.")
        generated_demand = np.random.poisson(lam=avg_demand, size=num_periods)
        
    elif dist_type == "Uniform":
        variation = st.sidebar.number_input("Range Variation (+/- from Mean)", min_value=1.0, value=30.0, step=5.0)
        low = max(0, avg_demand - variation)
        high = avg_demand + variation
        generated_demand = np.random.uniform(low=low, high=high, size=num_periods)

    # Create DataFrame
    df = pd.DataFrame({
        'Period': range(1, num_periods + 1),
        'Demand': np.round(generated_demand, 0)
    })

else:
    st.sidebar.subheader("Upload Excel File")
    
    # 📥 Provide a Sample Template Download
    sample_df = pd.DataFrame({
        'Period': ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5'],
        'Demand': [120, 95, 150, 80, 110]
    })
    
    # Buffer to hold excel data
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        sample_df.to_excel(writer, index=False, sheet_name='Sheet1')
    
    st.sidebar.download_button(
        label="📥 Download Sample Excel Template",
        data=buffer.getvalue(),
        file_name="demand_template.xlsx",
        mime="application/vnd.ms-excel"
    )
    
    uploaded_file = st.sidebar.file_path = st.sidebar.file_uploader("Upload your .xlsx file:", type=["xlsx"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            # Basic validation
            if 'Demand' not in df.columns:
                st.error("❌ The Excel file must contain a column named exactly **'Demand'**.")
                df = None
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("👋 Please upload an Excel file or switch to 'Generate Synthetic Data' in the sidebar.")

# --- MAIN PAGE: VISUALIZATION & ANALYSIS ---
if df is not None:
    
    # Layout Layout splits: Left for Metrics, Right for Histogram Settings
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("📋 Summary Stats")
        st.metric("Total Demand", f"{int(df['Demand'].sum()):,}")
        st.metric("Average Demand", f"{df['Demand'].mean():.1f}")
        st.metric("Max Demand", f"{int(df['Demand'].max())}")
        st.metric("Min Demand", f"{int(df['Demand'].min())}")
        
    with col2:
        st.subheader("🎛️ Histogram Settings")
        # Bin selection slider
        min_demand = int(df['Demand'].min())
        max_demand = int(df['Demand'].max())
        range_demand = max_demand - min_demand
        
        # Smart default for bins
        default_bins = min(30, max(5, int(range_demand / 5) if range_demand > 0 else 10))
        
        bins = st.slider("Select Number of Bins:", min_value=5, max_value=100, value=default_bins, step=1)
        
        # Plotting the histogram using Plotly for interactivity
        fig = px.histogram(
            df, 
            x="Demand", 
            nbins=bins,
            title="Demand Distribution Histogram",
            labels={'Demand': 'Demand Quantity', 'count': 'Frequency (Periods)'},
            color_discrete_sequence=['#1f77b4'],
            template="plotly_white"
        )
        
        fig.update_layout(
            bargap=0.05,
            yaxis_title="Frequency (How often it happened)",
            xaxis_title="Demand Level"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # Show Data Table option
    with st.expander("👀 View Raw Data Table"):
        st.dataframe(df, use_container_width=True)
