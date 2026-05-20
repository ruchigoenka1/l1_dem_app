import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import plotly.graph_objects as go
from prophet import Prophet
from scipy.stats import norm
import scipy.stats as stats

# --- Session State Initialization ---
if 'next_clicked' not in st.session_state:
    st.session_state.next_clicked = False
if 'seed_counter' not in st.session_state:
    st.session_state.seed_counter = 42

st.set_page_config(page_title="Supply Chain Analytics Platform", layout="wide")

st.title("🚀 Supply Chain Analytics Platform")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Average Demand", "📊 Demand Histogram", "📈 Demand Forecasting", "Demand Simulator Game", "Inventory Audit"])

with tab1:
    st.header("The Basic Thumb Rule Used For Inventory Planning")
    # st.markdown("""
    # **The Concept:** Demonstrating how static, average-based demand strategies systematically introduce internal sabotage. 
    # While an average looks clean over a 300-day window, daily variability will trigger stockouts during finite replenishment cycles.
    # """)
    
    # --- Step 1: Baseline Strategy Input Section ---
    col1, col2 = st.columns(2)
    
    with col1:
        annual_sales = st.number_input("Annual Sales (Units)", value=12000, step=500)
        working_days = st.number_input("Working Days per Year", value=300)
        
    with col2:
        # Calculate Average Daily Sales (ADS) baseline
        avg_daily_sales = annual_sales / working_days
        st.metric("Avg. Daily Sales (ADS)", f"{avg_daily_sales:.2f}")
        
        suggested_baseline = avg_daily_sales * 10
        requisite_inventory = st.number_input(
            "Enter Requisite Inventory Strategy Limit", 
            value=int(suggested_baseline),
            help="This is the target inventory volume you have allocated to cover your business lead time window."
        )

    # Trigger persistent UI view state
    if st.button("Next"):
        st.session_state.next_clicked = True

    # --- Step 2: Persisted Stress-Testing Environment ---
    if st.session_state.next_clicked:
        st.divider()
        st.subheader("🎯 Stress Test Parameters & Reality Simulator")
        
        # User Parameter Input Boxes
        c1, c2, c3 = st.columns(3)
        with c1:
            std_dev = st.number_input("Demand Standard Deviation (Volatility)", value=10, min_value=0)
        with c2:
            sim_days = st.number_input("Number of Simulation Days", value=100, min_value=1)
        with c3:
            rolling_window = st.number_input("Look-Forward Window (Days)", value=10, min_value=1, max_value=int(sim_days))

        # Action Buttons Layout: Regenerate Button
        btn_col1, btn_col2 = st.columns([1, 5])
        with btn_col1:
            if st.button("🔄 Regenerate Demand"):
                st.session_state.seed_counter += 1  # Shifts the seed to force a new layout run

        # Generate Volatile Demand Data Array
        np.random.seed(st.session_state.seed_counter)
        daily_demand = np.random.normal(avg_daily_sales, std_dev, sim_days)
        daily_demand = np.clip(daily_demand, 0, None).round(0)  # Prevents impossible negative demand days
        
        days = [f"Day {i+1}" for i in range(sim_days)]
        
        # --- Visual Asset 1: Daily Demand Timeline ---
        st.write("### 📈 Daily Demand Volatility")
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(
            x=days, y=daily_demand, mode='lines+markers', name='Daily Demand Actual',
            line=dict(color='#1f77b4', width=2)
        ))
        fig_daily.add_hline(y=avg_daily_sales, line_dash="dash", line_color="gray", annotation_text="Calculated Static Average")
        fig_daily.update_layout(template="plotly_white", height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_daily, use_container_width=True)

        # Pre-calculating Data for Tables & Charts
        df_summary = pd.DataFrame({
            "Lead Time Day": days,
            "Daily Demand (Units)": daily_demand.astype(int)
        })
        
        # Look-Forward Core Mathematical Optimization Matrix
        forward_sums = df_summary["Daily Demand (Units)"].iloc[::-1].rolling(window=rolling_window).sum().iloc[::-1]
        df_summary[f"Demand Next {rolling_window} Days"] = forward_sums
        df_summary["Inventory Level Provided"] = int(requisite_inventory)
        
        # Metric Scorecard Data Compilation
        valid_forward_days = forward_sums.dropna()
        total_valid_days = len(valid_forward_days)
        deficits_series = valid_forward_days > requisite_inventory
        total_deficits = deficits_series.sum()
        pct_deficits = (total_deficits / total_valid_days * 100) if total_valid_days > 0 else 0.0
        
        # Calculate Maximum Forward Window Value
        max_window_demand = valid_forward_days.max() if total_valid_days > 0 else 0.0

        # --- Visual Asset 2: Collapsible Diagnostic Data Table & Scorecard ---
        with st.expander("📋 Generated Demand Data Table", expanded=False):
            st.markdown("### 📊 Window Analysis Summary")
            
            # Expanded layout matrix (changed to 4 columns to fit the new metric)
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Days with Valid Window", f"{total_valid_days} Days")
            with m2:
                # Calculate absolute peak gap to show if the strategy safely absorbed it
                peak_gap = int(max_window_demand - requisite_inventory)
                st.metric(
                    "Max Window Demand Peak", 
                    f"{int(max_window_demand)} Units",
                    delta=f"+{peak_gap} Over Limit" if peak_gap > 0 else f"{peak_gap} Under Limit",
                    delta_color="inverse" if peak_gap > 0 else "normal"
                )
            with m3:
                st.metric("Total Deficit Occurrences", f"{total_deficits} Days", 
                          delta=f"-{total_deficits} Stockouts" if total_deficits > 0 else None, 
                          delta_color="inverse" if total_deficits > 0 else "normal")
            with m4:
                st.metric("Deficit Risk Rate (%)", f"{pct_deficits:.1f}%",
                          delta="CRITICAL RISK" if pct_deficits > 30 else "STABLE BUFFER",
                          delta_color="inverse" if pct_deficits > 30 else "normal")
                
            st.divider()

            # Row-by-row functional mapper for color injection logic
            def calculate_status(row):
                forward_demand = row[f"Demand Next {rolling_window} Days"]
                if pd.isna(forward_demand):
                    return ""
                
                net_value = int(row["Inventory Level Provided"] - forward_demand)
                if net_value >= 0:
                    return f'<span style="color: #2e7d32; font-weight: bold;">🟢 Surplus (+{net_value})</span>'
                else:
                    return f'<span style="color: #d32f2f; font-weight: bold;">🔴 Deficit ({net_value})</span>'

            # Build and finalize table display dataframe
            df_table = df_summary.copy()
            df_table["Net Status"] = df_table.apply(calculate_status, axis=1)
            df_table[f"Demand Next {rolling_window} Days"] = df_table[f"Demand Next {rolling_window} Days"].apply(
                lambda x: f"{int(x)}" if not pd.isna(x) else ""
            )
            
            st.write(df_table.to_html(escape=False, index=False), unsafe_allow_html=True)
            st.write("<br>", unsafe_allow_html=True)

        # --- Visual Asset 3: Collapsible Charts for Forward Window Analytics ---
        with st.expander("📊 View Forward Window Trend & Distribution Analysis", expanded=False):
            df_clean_charts = df_summary.dropna().copy()
            
            graph_col1, graph_col2 = st.columns(2)
            
            with graph_col1:
                st.markdown(f"#### 📉 Forward Window Demand Trend")
                fig_trend = go.Figure()
                
                fig_trend.add_trace(go.Scatter(
                    x=df_clean_charts["Lead Time Day"], 
                    y=df_clean_charts[f"Demand Next {rolling_window} Days"],
                    mode='lines',
                    name=f'{rolling_window}-Day Demand',
                    line=dict(color='#1f77b4', width=2)
                ))
                fig_trend.add_hline(
                    y=requisite_inventory, 
                    line_dash="dash", 
                    line_color="#d62728", 
                    annotation_text="Your Stock Limit",
                    annotation_position="top left"
                )
                fig_trend.update_layout(
                    template="plotly_white", 
                    xaxis_title="Simulation Day Index",
                    yaxis_title="Total Window Units",
                    height=350,
                    margin=dict(t=30, b=10)
                )
                st.plotly_chart(fig_trend, use_container_width=True)
                
            with graph_col2:
                st.markdown(f"#### 📊 Look-Forward Window Distribution")
                
                fig_hist = px.histogram(
                    df_clean_charts, 
                    x=f"Demand Next {rolling_window} Days",
                    nbins=20,
                    color_discrete_sequence=['#1f77b4']
                )
                fig_hist.add_vline(
                    x=requisite_inventory, 
                    line_dash="dash", 
                    line_color="#d62728", 
                    annotation_text="Stock Ceiling",
                    annotation_position="top right"
                )
                fig_hist.update_layout(
                    template="plotly_white",
                    xaxis_title=f"Aggregated Demand in {rolling_window}-Day Windows",
                    yaxis_title="Frequency Occurrence Count",
                    height=350,
                    margin=dict(t=30, b=10),
                    showlegend=False
                )
                st.plotly_chart(fig_hist, use_container_width=True)

        # Final Summary Executive Alerts
        if total_deficits > 0:
            st.error(f"❌ **Internal Sabotage Confirmed:** Volatility breached your static 'Average' allocation baseline strategy on **{total_deficits} separate window cycles** ({pct_deficits:.1f}% risk rate).")
        else:
            st.success(f"✅ **Strategic Parameter Verified.** Under these isolated settings, the current allocation buffer safely absorbed the simulated variance across all window blocks.")



with tab2:
    st.header("Demand Histogram Analyzer")
    
    # --- 1. Data Configuration ---
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
            generated = np.random.uniform(avg_demand - variation, avg_demand + variation, num_periods)
        
        df = pd.DataFrame({'Demand': np.floor(np.clip(generated, 0, None))})

    elif data_source == "Upload Your Own Data":
        up_col1, up_col2 = st.columns([2, 1])
        
        with up_col1:
            uploaded_file = st.file_uploader("Upload your historical demand file (.xlsx or .csv):", type=["xlsx", "csv"])
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df_upload = pd.read_csv(uploaded_file)
                    else:
                        df_upload = pd.read_excel(uploaded_file)
                    
                    if 'Demand' in df_upload.columns:
                        df = df_upload[['Demand']].dropna().copy()
                        df['Demand'] = pd.to_numeric(df['Demand'], errors='coerce')
                        df = df.dropna()
                        st.success("✅ File successfully uploaded and parsed!")
                    else:
                        st.error("❌ Invalid Format: Your file must contain a column named exactly **'Demand'**.")
                except Exception as e:
                    st.error(f"❌ Error loading file: {e}")
                    
        with up_col2:
            st.markdown("#### 📋 Download Template")
            st.caption("Please match your data format to this template. The sheet must include a column header named **Demand**.")
            
            template_df = pd.DataFrame({'Demand': [120, 95, 110, 135, 80, 105, 115]})
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                template_df.to_excel(writer, index=False, sheet_name='Template')
            
            st.download_button(
                label="📥 Download Excel Template",
                data=buffer.getvalue(),
                file_name="demand_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    # --- Collapsible Raw Data Table ---
    if df is not None:
        with st.expander("🔢 View / Download Raw Data Table", expanded=False):
            raw_display_df = df.copy()
            raw_display_df.index.name = "Period"
            
            exp_col1, exp_col2 = st.columns([3, 1])
            with exp_col1:
                st.dataframe(raw_display_df, use_container_width=True, height=250)
            with exp_col2:
                st.markdown("#### Export Current Data")
                st.caption("Download this active dataset as a CSV file for offline use.")
                csv_data = raw_display_df.to_csv(index=True).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name="demand_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    # --- 2. Advanced Analysis (Thresholds & Percentiles) ---
    if df is not None:
        st.divider()
        st.subheader("2. Probability & Coverage Analysis")
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            st.markdown("#### Threshold Lookup (Points Below X)")
            threshold = st.number_input("Enter Demand Value:", value=40.0, step=1.0)
            count_below = len(df[df['Demand'] < threshold])
            percent_below = (count_below / len(df)) * 100
            st.metric(f"Chances of Demand < {threshold}", f"{percent_below:.1f}%")
            st.caption(f"There are {count_below} periods where demand was less than {threshold}.")

        with analysis_col2:
            st.markdown("#### Percentile Lookup (Coverage Level)")
            target_perc = st.number_input("Enter Service Level % (e.g. 95):", min_value=0.0, max_value=100.0, value=95.0, step=1.0)
            demand_at_perc = np.percentile(df['Demand'], target_perc)
            st.metric(f"Demand at {target_perc}% Service Level", f"{int(demand_at_perc)}")
            st.caption(f"To cover {target_perc}% of all periods, you need to satisfy a demand of {int(demand_at_perc)}.")

        # --- 3. Visual Distribution & Tables Below ---
        st.divider()
        st.subheader("3. Visual Distribution")
        
        num_bins = st.slider("Select Number of Bins:", 5, 50, 15)
        
        counts, bin_edges = np.histogram(df['Demand'], bins=num_bins)
        bin_size = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 1

        fig = px.histogram(df, x="Demand", template="plotly_white", color_discrete_sequence=['#4F8BF9'])
        
        fig.update_traces(
            xbins=dict(
                start=bin_edges[0],
                end=bin_edges[-1],
                size=bin_size
            )
        )
        
        fig.add_vline(
            x=threshold, 
            line_dash="dot", 
            line_color="#EF553B", 
            line_width=2.5,
            annotation_text=f"Threshold ({threshold})", 
            annotation_position="top left"
        )
        
        fig.add_vline(
            x=demand_at_perc, 
            line_dash="dot", 
            line_color="#00CC96", 
            line_width=2.5,
            annotation_text=f"{target_perc}% Service Level ({int(demand_at_perc)})", 
            annotation_position="top right"
        )
        
        fig.update_layout(bargap=0.1, xaxis_title="Demand Quantity", yaxis_title="Count of Periods")
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        table_col1, table_col2 = st.columns([1, 1])

        with table_col1:
            st.markdown("#### 📋 Statistical Summary")
            summary_stats = df['Demand'].describe().to_frame().T
            st.dataframe(summary_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']], use_container_width=True)

        with table_col2:
            st.markdown("#### Bin Frequency Table")
            pct_total = counts / len(df) * 100
            
            bin_df = pd.DataFrame({
                "Bin Range": [f"{int(bin_edges[i])} - {int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)],
                "Frequency (Count)": counts,
                "% of Total": pct_total.round(1),
                "Cum. Count": counts.cumsum(),
                "Cum. %": pct_total.cumsum().round(1)
            })
            st.dataframe(bin_df, use_container_width=True, hide_index=True)

        # --- 4. Coefficient of Variation (CoV) Analysis ---
        st.divider()
        st.subheader("📊 Demand Volatility Analysis (CoV)")
        
        cov_col1, cov_col2 = st.columns([1, 2])
        
        with cov_col1:
            st.markdown("#### Formula")
            st.latex(r"CoV = \frac{\sigma}{\mu}")
            st.caption(r"Where $\sigma$ = Standard Deviation and $\mu$ = Mean")
            
        with cov_col2:
            # Extract statistics directly from data stream
            mean_val = float(df['Demand'].mean())
            std_val = float(df['Demand'].std())
            
            # Defensive check for edge case where mean is zero
            cov_val = (std_val / mean_val) if mean_val > 0 else 0.0
            
            # Determine demand volatility profile category
            if cov_val <= 0.10:
                status_text = "🟢 Ultra-Stable / Constant"
                explanation = "Highly repetitive and predictable demand. Use automated just-in-time (JIT) scheduling or lean kanbans. Minimize safety stock to release working capital."
                alert_type = "success"
            elif cov_val <= 0.25:
                status_text = "🟢 Stable / Predictable"
                explanation = "Normal variation patterns present. Standard statistical forecasting and fixed reorder points will yield high accuracy with minimal safety stock buffers."
                alert_type = "success"
            elif cov_val <= 0.50:
                status_text = "🟡 Moderate Volatility"
                explanation = "Demand exhibits noticeable fluctuations. Requires proactive demand sensing and traditional statistical safety stocks to counter stockout risks."
                alert_type = "warning"
            elif cov_val <= 1.00:
                status_text = "🟠 High Volatility"
                explanation = "Highly variable demand spikes. Avoid automated ordering systems without collaborative forecasting inputs. Expect to maintain higher, dynamic safety stock thresholds."
                alert_type = "warning"
            else:
                status_text = "🔴 Erratic / Lumpy / Sporadic"
                explanation = "Highly unpredictable or intermittent demand. Traditional safety stock formulas do not work well here. Consider move-to-order (MTO) execution or project-based buffers."
                alert_type = "error"
                
            # Render key calculation metrics inside columns
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Mean ($\mu$)", f"{mean_val:.2f}")
            with m_col2:
                st.metric("Std Dev ($\sigma$)", f"{std_val:.2f}")
            with m_col3:
                st.metric("Calculated CoV", f"{cov_val:.3f}")
                
            # Render descriptive behavioral classification banner
            st.markdown(f"### Profile: {status_text}")
            st.info(explanation)

# Placeholder layouts for future tabs
with tab3:
    st.header("🧬 Stage 3: The Probability Truth")
    st.markdown("Analyze historical patterns, simulate growth, and project future demand with AI uncertainty bands.")
    
    # --- 1. DATA SOURCE & SAMPLE DOWNLOAD ---
    col_header, col_download = st.columns([2, 1])
    with col_header:
        data_mode = st.radio("Data Mode:", ("Simulation", "Upload Data"), horizontal=True, key="mode_t3")
    
    with col_download:
        # Create a sample template with growth and seasonality
        sample_dates = pd.date_range(start="2024-01-01", periods=365, freq='D')
        t_sample = np.arange(365)
        sample_y = 500 + (15 * t_sample/30) + (50 * np.sin(2 * np.pi * t_sample / 30)) + np.random.normal(0, 30, 365)
        sample_df = pd.DataFrame({'Date': sample_dates, 'Demand': np.maximum(0, sample_y).astype(int)})
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            sample_df.to_excel(writer, index=False)
        
        st.download_button(
            label="📥 Download Workshop Template.xlsx",
            data=buffer.getvalue(),
            file_name="demand_template.xlsx",
            mime="application/vnd.ms-excel"
        )

    df_truth = None

    if data_mode == "Simulation":
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                level = st.number_input("Base Level (Start)", value=100.0)
                growth = st.number_input("Annual Growth %", value=15.0) # Increased for visible 'cone'
            with c2:
                base_amp = st.number_input("Amplitude", value=400.0)
                target_cov = st.number_input("Target CoV", value=0.15)
            with c3:
                surcharge = st.slider("Peak Surcharge %", 0, 100, 30)
            with c4:
                forecast_days = st.number_input("Forecast Horizon (Days)", value=365)

        # SIMULATION GENERATION
        dates = pd.date_range(start="2023-01-01", periods=730, freq='D')
        t = np.arange(len(dates))
        growth_factor = (1 + growth/100) ** (t / 365)
        seasonal_wave = np.sin(2 * np.pi * t / 365.25)
        
        baseline_calc = level + base_amp
        y_vals = (baseline_calc + (seasonal_wave * (base_amp * 0.5))) * growth_factor
        y_vals += np.random.normal(0, (base_amp * target_cov), len(dates))
        
        df_truth = pd.DataFrame({'ds': dates, 'y': np.maximum(0, y_vals)})
        
        # Labeling and Surcharge
        high_t = baseline_calc * 1.25
        low_t = baseline_calc * 0.75
        df_truth.loc[df_truth['y'] > high_t, 'y'] *= (1 + surcharge/100)
        df_truth['Seasonality'] = df_truth['y'].apply(lambda x: 'High' if x > high_t else ('Low' if x < low_t else 'Normal'))

    else:
        uploaded_file = st.file_uploader("Upload xlsx", type=["xlsx"], key="up_t3")
        forecast_days = st.number_input("Forecast Horizon (Days)", value=180)
        if uploaded_file:
            df_truth = pd.read_excel(uploaded_file).rename(columns={'Date':'ds', 'Demand':'y'})
            df_truth['ds'] = pd.to_datetime(df_truth['ds'])
            q1, q3 = df_truth['y'].quantile([0.25, 0.75])
            df_truth['Seasonality'] = df_truth['y'].apply(lambda x: 'High' if x > q3 else ('Low' if x < q1 else 'Normal'))

    # --- 2. THE SEASONAL METRICS MATRIX ---
    if df_truth is not None:
        st.divider()
        st.subheader("📊 The Seasonal Metrics Matrix")
        
        # Calculate stats
        matrix = df_truth.groupby('Seasonality')['y'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
        matrix['CoV'] = (matrix['std'] / matrix['mean']).round(3)
        matrix.columns = ['Season', 'Avg Demand', 'Std Dev', 'Min', 'Max', 'Days Count', 'CoV']
        
        # Display styled table
        st.dataframe(
            matrix.style.background_gradient(subset=['CoV'], cmap='RdYlGn_r').format(precision=2),
            use_container_width=True, hide_index=True
        )

        # --- 3. DUAL HISTOGRAMS ---
        col_l, col_r = st.columns(2)
        with col_l:
            st.plotly_chart(px.histogram(df_truth, x="y", title="A. General Distribution", template="plotly_dark", nbins=40), use_container_width=True)
        with col_r:
            fig_s = px.histogram(df_truth, x="y", color="Seasonality", title="B. Seasonal Breakdown", template="plotly_dark", barmode='overlay',
                                 color_discrete_map={"Normal": "#5B84B1", "High": "#FC766A", "Low": "#71918d"})
            st.plotly_chart(fig_s, use_container_width=True)

        # --- 4. PROPHET FORECAST (With Broadening Uncertainty) ---
        st.divider()
        st.subheader("🔮 AI Prophet Forecast (Trend & Uncertainty)")
        
        with st.spinner("AI training and calculating risk bands..."):
            # Changepoint_prior_scale=0.5 makes the uncertainty cone widen significantly
            m = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.5)
            m.fit(df_truth)
            
            future = m.make_future_dataframe(periods=int(forecast_days))
            forecast = m.predict(future)
            
            fig_f = go.Figure()

            # The broadening Uncertainty Ribbon
            fig_f.add_trace(go.Scatter(
                x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
                fill='toself', fillcolor='rgba(100, 100, 100, 0.3)', line=dict(color='rgba(255,255,255,0)'),
                name='Uncertainty Interval (95%)'
            ))

            # AI Prediction Line
            fig_f.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], line=dict(color='#4F8BF9', width=3), name='AI Forecast'))

            # Actual Historical Points
            fig_f.add_trace(go.Scatter(x=df_truth['ds'], y=df_truth['y'], mode='markers', marker=dict(color='white', size=2), name='Actual Data'))

            fig_f.update_layout(
                template="plotly_dark",
                title=f"Prophet Projection for {forecast_days} Days",
                yaxis=dict(range=[0, forecast['yhat_upper'].max() * 1.1]), # Force Y-axis to start at 0
                xaxis_title="Date", yaxis_title="Demand Quantity",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_f, use_container_width=True)
            
        with st.expander("📂 View Raw Data Table"):
            st.dataframe(df_truth, use_container_width=True)

# Assuming your layout structure looks like this:
# tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Demand Analyzer", "Audit Log", "Safety Stock Game"])

with tab4:
    st.header("🎯 Tab 4: Safety Stock Simulation Game")
    st.markdown("""
    **The Challenge:** Balance inventory holding costs against shortages. 
    Replenishment orders are sent at the **end of the day** and arrive on the **morning of Day L+1**.
    """)

    # =========================================================================
    # GLOBAL VISUAL STYLE LAYOUT CONFIGURATION
    # =========================================================================
    shared_layout = dict(
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0", family="sans-serif"), 
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.07)", zeroline=False, linecolor="rgba(255, 255, 255, 0.15)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.07)", zeroline=False, linecolor="rgba(255, 255, 255, 0.15)")
    )

    # =========================================================================
    # SECTION 1: DATA CONFIGURATION
    # =========================================================================
    st.markdown("### 1. Data Configuration")
    
    dist_type = st.selectbox("Distribution Type", ["Uniform", "Normal"], index=0, key="t4_dist_type")
    
    col_cfg1, col_cfg2 = st.columns(2)

    with col_cfg1:
        avg_demand = st.number_input("Average Daily Demand", min_value=1, value=50, step=5, key="t4_avg_dem")
        
    with col_cfg2:
        if dist_type == "Uniform":
            variation = st.number_input("Daily Variation (± From Average)", min_value=0, value=25, step=5, key="t4_variation")
            low_bound = max(0, avg_demand - variation)
            high_bound = avg_demand + variation
            std_dev = 0
        else:
            std_dev = st.number_input("Daily Std Dev (σ)", min_value=0.0, value=10.0, step=1.0, key="t4_std_dev")
            low_bound, high_bound, variation = 0, 0, 0

    if dist_type == "Uniform":
        st.markdown(f"🔹 *Daily Range: **{low_bound}** to **{high_bound}** units*")

    st.markdown("### 2. Backlog & Stockout Policy")
    col_back1, col_back2 = st.columns(2)
    
    with col_back1:
        allow_backlog = st.radio("Can unfulfilled demand be backlogged for later?", ["Yes", "No"], index=0, key="t4_allow_backlog")
        
    with col_back2:
        if allow_backlog == "Yes":
            backlog_limit = st.number_input("Maximum Backlog Capacity Limit (0 for Unlimited)", min_value=0, value=100, step=10, key="t4_backlog_limit")
        else:
            st.markdown("<div style='padding-top: 30px; color: gray; font-size: 14px;'>❌ Lost Sales Policy Active: Inventory cannot drop below 0.</div>", unsafe_allow_html=True)
            backlog_limit = 0

    # =========================================================================
    # SECTION 2: COLLAPSIBLE REORDER POINT (ROP) CALCULATOR & GRAPH
    # =========================================================================
    st.markdown("---")
    
    # Wrapped entirely inside a collapsible tray container
    with st.expander("📈 View Reorder Point (ROP) Analytics & Lead Time Math", expanded=False):
        col_rop1, col_rop2 = st.columns([1.2, 1.8])
        
        with col_rop1:
            st.markdown("**ROP Target Parameters**")
            lead_time = st.number_input("Supplier Lead Time (Days)", min_value=1, value=3, step=1, key="t4_lt")
            target_service_level = st.slider("Target Service Level (%)", min_value=50.0, max_value=99.9, value=95.0, step=0.5, key="t4_tsl")
            
            # Lead Time Statistical Scaling Math
            lt_avg_demand = avg_demand * lead_time
            
            if dist_type == "Normal":
                lt_std_dev = std_dev * np.sqrt(lead_time)
                z_score = stats.norm.ppf(target_service_level / 100.0) if lt_std_dev > 0 else 0
                safety_stock = int(np.ceil(z_score * lt_std_dev))
                calculated_rop = int(np.ceil(lt_avg_demand + safety_stock))
            else:
                sim_rng = np.random.RandomState(42)
                if variation == 0:
                    lt_samples = np.full(10000, lt_avg_demand)
                else:
                    lt_samples = np.sum(sim_rng.randint(low_bound, high_bound + 1, size=(lead_time, 10000)), axis=0)
                
                calculated_rop = int(np.percentile(lt_samples, target_service_level))
                safety_stock = max(0, calculated_rop - lt_avg_demand)

            st.markdown("#### **Calculation Results**")
            st.metric("Expected Lead Time Demand", f"{int(lt_avg_demand)} units")
            st.metric("Required Safety Stock Buffer", f"{int(safety_stock)} units")
            st.metric("Suggested Reorder Point (ROP)", f"{int(calculated_rop)} units")

        with col_rop2:
            st.markdown(f"**Total Demand Distribution Over {lead_time}-Day Lead Time Window**")
            
            if dist_type == "Normal":
                if lt_std_dev > 0:
                    x_axis_range = np.linspace(lt_avg_demand - 4*lt_std_dev, lt_avg_demand + 4*lt_std_dev, 200)
                    y_axis_density = stats.norm.pdf(x_axis_range, lt_avg_demand, lt_std_dev)
                else:
                    x_axis_range = np.array([lt_avg_demand - 5, lt_avg_demand, lt_avg_demand + 5])
                    y_axis_density = np.array([0, 1, 0])
                
                fig_rop_dist = go.Figure()
                fig_rop_dist.add_trace(go.Scatter(x=x_axis_range, y=y_axis_density, mode='lines', line=dict(color='#A370F7', width=3), name='Probability Density', fill='tozeroy', fillcolor='rgba(163, 112, 247, 0.1)'))
            else:
                counts, bins = np.histogram(lt_samples, bins='auto', density=True)
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                fig_rop_dist = go.Figure()
                fig_rop_dist.add_trace(go.Scatter(x=bin_centers, y=counts, mode='lines', line=dict(color='#A370F7', width=3, shape='spline'), name='Compounded Shape', fill='tozeroy', fillcolor='rgba(163, 112, 247, 0.1)'))

            fig_rop_dist.add_vline(x=lt_avg_demand, line_width=2, line_dash="dash", line_color="#3A96FF", annotation_text="Expected Demand", annotation_position="top left")
            fig_rop_dist.add_vline(x=calculated_rop, line_width=2.5, line_color="#FF5A5A", annotation_text=f"ROP ({target_service_level}%)", annotation_position="top right")
            
            fig_rop_dist.update_layout(
                **shared_layout,
                showlegend=False, 
                height=280
            )
            fig_rop_dist.update_layout(margin=dict(l=30, r=30, t=20, b=30))
            fig_rop_dist.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
            st.plotly_chart(fig_rop_dist, use_container_width=True)

    # Added fallback logic if expander hasn't run yet to prevent initialization crashes
    if 'calculated_rop' not in locals():
        calculated_rop = 150
        lead_time = 3

    with st.expander("🛠️ Advanced Asset Deployment Settings", expanded=False):
        col_inv1, col_inv2, col_inv3 = st.columns(3)
        with col_inv1:
            reorder_point = st.number_input("Reorder Point (ROP)", min_value=0, value=calculated_rop, step=10, key="t4_rop")
        with col_inv2:
            starting_inventory = st.number_input("Starting On-Hand Inventory", min_value=1, value=188, step=10, key="t4_start_inv")
        with col_inv3:
            order_qty = st.number_input("Replenishment Batch Size (Q)", min_value=1, value=200, step=10, key="t4_q")

    # =========================================================================
    # SECTION 3: CORE SIMULATION ENGINE
    # =========================================================================
    if 't4_history' not in st.session_state:
        st.session_state.t4_history = pd.DataFrame(columns=[
            'Day', 'Opening Stock', 'Arrived Morning', 'Updated Opening Stock', 
            'Demand Generated', 'Sales Met', 'Shortage', 'Unfulfilled Backlog', 
            'Closing Inventory', 'Order Placed Evening', 'Total Pipeline Inventory', 'Pipeline Status'
        ])
        st.session_state.t4_day_counter = 0
        st.session_state.t4_current_inv = starting_inventory
        st.session_state.t4_backlog = 0  
        st.session_state.t4_pipeline_orders = [] 

    def run_simulation_steps(num_days):
        if 't4_backlog' not in st.session_state:
            st.session_state.t4_backlog = 0
            
        history_df = st.session_state.t4_history.copy()
        day_counter = st.session_state.t4_day_counter
        current_inv = st.session_state.t4_current_inv
        backlog = st.session_state.t4_backlog
        pipeline_orders = list(st.session_state.t4_pipeline_orders)
        
        rng = np.random.RandomState()
        new_records = []

        for _ in range(num_days):
            day_counter += 1
            initial_opening_stock = current_inv
            
            # Morning Arrivals
            arriving_qty = sum(order['qty'] for order in pipeline_orders if order['delivery_day'] == day_counter)
            pipeline_orders = [order for order in pipeline_orders if order['delivery_day'] != day_counter]
            
            if arriving_qty > 0:
                if allow_backlog == "Yes" and backlog > 0:
                    if arriving_qty >= backlog:
                        arriving_qty -= backlog
                        backlog = 0
                    else:
                        backlog -= arriving_qty
                        arriving_qty = 0
                current_inv += arriving_qty
            
            updated_opening_stock = current_inv
            
            # Demand Gen
            if dist_type == "Normal":
                demand = max(0, int(rng.normal(float(avg_demand), float(std_dev))))
            else:
                if low_bound == high_bound:
                    demand = int(avg_demand)
                else:
                    demand = int(rng.randint(int(low_bound), int(high_bound) + 1))
            
            # Fulfill Demand
            total_needed = demand + backlog
            if updated_opening_stock >= total_needed:
                sales_met = demand
                shortage = 0
                backlog = 0
                closing_inv = updated_opening_stock - total_needed
            else:
                sales_met = max(0, updated_opening_stock - backlog)
                raw_shortage = total_needed - updated_opening_stock
                
                if allow_backlog == "Yes":
                    backlog = min(raw_shortage, backlog_limit) if backlog_limit > 0 else raw_shortage
                    shortage = raw_shortage
                else:
                    backlog = 0
                    shortage = demand - updated_opening_stock
                    
                closing_inv = 0
                
            # Evening Order Placements
            pipeline_qty_before_order = sum(order['qty'] for order in pipeline_orders)
            inventory_position = closing_inv + pipeline_qty_before_order - backlog
            
            order_placed_tonight = 0
            if inventory_position <= reorder_point:
                order_placed_tonight = order_qty
                target_delivery = day_counter + lead_time + 1
                pipeline_orders.append({'delivery_day': target_delivery, 'qty': order_qty})
            
            total_pipeline_inventory = sum(order['qty'] for order in pipeline_orders)
            
            if order_placed_tonight > 0:
                pipeline_status = f"Placed Order (Arriving Day {target_delivery} Morning)"
            elif total_pipeline_inventory > 0:
                pipeline_status = f"{total_pipeline_inventory} units en route"
            else:
                pipeline_status = "Clear"

            new_records.append({
                'Day': day_counter, 'Opening Stock': initial_opening_stock, 'Arrived Morning': arriving_qty,
                'Updated Opening Stock': updated_opening_stock, 'Demand Generated': demand, 'Sales Met': sales_met,
                'Shortage': shortage, 'Unfulfilled Backlog': backlog, 'Closing Inventory': closing_inv,
                'Order Placed Evening': order_placed_tonight, 'Total Pipeline Inventory': total_pipeline_inventory,
                'Pipeline Status': pipeline_status
            })
            
            current_inv = closing_inv

        if new_records:
            new_df = pd.DataFrame(new_records)
            st.session_state.t4_history = pd.concat([history_df, new_df], ignore_index=True)
            
        st.session_state.t4_day_counter = day_counter
        st.session_state.t4_current_inv = current_inv
        st.session_state.t4_backlog = backlog
        st.session_state.t4_pipeline_orders = pipeline_orders

    if st.button("🔄 Reset Simulation Data", key="t4_reset_btn"):
        st.session_state.t4_history = pd.DataFrame(columns=[
            'Day', 'Opening Stock', 'Arrived Morning', 'Updated Opening Stock', 
            'Demand Generated', 'Sales Met', 'Shortage', 'Unfulfilled Backlog', 
            'Closing Inventory', 'Order Placed Evening', 'Total Pipeline Inventory', 'Pipeline Status'
        ])
        st.session_state.t4_day_counter = 0
        st.session_state.t4_current_inv = starting_inventory
        st.session_state.t4_backlog = 0
        st.session_state.t4_pipeline_orders = []
        st.rerun()

    st.markdown("---")

    # =========================================================================
    # SECTION 4: SIMULATION INTERACTION BUTTONS
    # =========================================================================
    st.subheader("🕹️ Simulation Actions")
    col_btn1, col_btn2 = st.columns([1, 1.5])

    with col_btn1:
        if st.button("☀️ Next Day (Single Step)", use_container_width=True, key="t4_step_btn"):
            run_simulation_steps(1)

    with col_btn2:
        sim_days = st.number_input("Days to fast-forward", min_value=2, max_value=365, value=30, step=5, label_visibility="collapsed", key="t4_sim_days_input")
        if st.button(f"⏩ Simulate {sim_days} Days", use_container_width=True, key="t4_batch_btn"):
            run_simulation_steps(sim_days)

    # =========================================================================
    # SECTION 5: REAL-TIME ANALYTICS LEDGERS
    # =========================================================================
    if not st.session_state.t4_history.empty:
        df = st.session_state.t4_history
        
        total_shortages = df['Shortage'].sum()
        stockout_days = int((df['Shortage'] > 0).sum())
        service_level = (df['Sales Met'].sum() / df['Demand Generated'].sum()) * 100 if df['Demand Generated'].sum() > 0 else 100

        st.markdown("---")
        st.subheader("📊 Live Performance Scoreboard")
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Current Day", int(df['Day'].iloc[-1]))
        m2.metric("Closing Stock", f"{int(df['Closing Inventory'].iloc[-1])} units")
        m3.metric("Service Level", f"{service_level:.1f}%")
        m4.metric("Pipeline Stock", f"{int(df['Total Pipeline Inventory'].iloc[-1])} units")
        m5.metric("Stock Out Days", f"{stockout_days} days", delta=f"{int(total_shortages)} units missed", delta_color="inverse")

        st.subheader("📈 Real-Time Tracking Analytics")
        col_graph1, col_graph2 = st.columns(2)

        with col_graph1:
            st.markdown("**Inventory Tracking Over Time**")
            net_inventory_curve = df['Closing Inventory'] - df['Unfulfilled Backlog']
            
            fig_inv = go.Figure()
            fig_inv.add_trace(go.Scatter(
                x=df['Day'], y=net_inventory_curve, mode='lines+markers', name='Net Inventory State',
                line=dict(color='#3A96FF', width=2.5, shape='spline'), marker=dict(size=5, color='#3A96FF'),
                fill='tozeroy', fillcolor='rgba(58, 150, 255, 0.1)'
            ))
            fig_inv.add_trace(go.Scatter(
                x=df['Day'], y=[reorder_point]*len(df), mode='lines', name='Reorder Point Target (ROP)',
                line=dict(color='#FF5A5A', width=2, dash='dash')
            ))
            fig_inv.update_layout(**shared_layout, xaxis_title="Day", yaxis_title="Units State Balance", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            
            lowest_point = min(net_inventory_curve.min() - 20, -20)
            highest_point = max(df['Closing Inventory'].max() + 20, reorder_point + 20)
            fig_inv.update_yaxes(range=[lowest_point, highest_point])
            st.plotly_chart(fig_inv, use_container_width=True)

        is_zero_variation = (dist_type == "Uniform" and variation == 0) or (dist_type == "Normal" and std_dev == 0.0)

        with col_graph2:
            st.markdown("**Generated Demand Distribution**")
            fig_hist = go.Figure()
            
            if is_zero_variation:
                fig_hist.add_trace(go.Bar(
                    x=[avg_demand], y=[len(df)], name='Demand Frequency',
                    marker=dict(color='rgba(58, 150, 255, 0.4)', line=dict(color='#3A96FF', width=1.5)),
                    width=[4.0]
                ))
                fig_hist.update_layout(**shared_layout, bargap=0.08, yaxis_title="Days Logged", showlegend=False)
                fig_hist.update_xaxes(range=[avg_demand - 10, avg_demand + 10], tickvals=[avg_demand], title_text="Demand Bracket")
            else:
                if dist_type == "Uniform":
                    total_elements = high_bound - low_bound + 1
                    bin_size = 5 if total_elements % 5 == 0 else (10 if total_elements % 10 == 0 else max(1, total_elements // 5))
                    breaks = np.arange(low_bound - 0.5, high_bound + 0.5 + bin_size, bin_size)
                else:
                    breaks = np.histogram_bin_edges(df['Demand Generated'], bins='sturges')
                
                fig_hist.add_trace(go.Histogram(
                    x=df['Demand Generated'], xbins=dict(start=breaks[0], end=breaks[-1], size=(breaks[1] - breaks[0])),
                    autobinx=False, name='Demand Frequency',
                    marker=dict(color='rgba(58, 150, 255, 0.4)', line=dict(color='#3A96FF', width=1.5))
                ))
                fig_hist.update_layout(**shared_layout, bargap=0.08, xaxis_title="Demand Bracket", yaxis_title="Days Logged", showlegend=False)
                
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")

        # COLLAPSIBLE TABLE 1: Distribution Table
        with st.expander("📊 View Distribution Bin Analysis Data Table", expanded=False):
            if is_zero_variation:
                bin_records = [{
                    "Demand Bracket Range": f"{avg_demand} to {avg_demand} units (Static)",
                    "Days Sampled (Count)": int(len(df)),
                    "Distribution Share (%)": "100.0%"
                }]
            else:
                counts, edges = np.histogram(df['Demand Generated'], bins=breaks)
                total_elements_count = len(df)
                bin_records = []
                for i in range(len(counts)):
                    lower_lbl = int(np.ceil(edges[i]))
                    upper_lbl = int(np.floor(edges[i+1]))
                    if dist_type == "Uniform" and (upper_lbl < low_bound or lower_lbl > high_bound):
                        continue
                    pct_share = (counts[i] / total_elements_count) * 100
                    bin_records.append({
                        "Demand Bracket Range": f"{lower_lbl} to {upper_lbl} units",
                        "Days Sampled (Count)": int(counts[i]),
                        "Distribution Share (%)": f"{pct_share:.1f}%"
                    })
            st.dataframe(pd.DataFrame(bin_records), use_container_width=True, hide_index=True)

        # COLLAPSIBLE TABLE 2: Operations Ledger
        with st.expander("📋 View Full Operations Ledger History Log", expanded=False):
            display_df = df.copy().sort_values(by='Day', ascending=False)
            st.dataframe(
                display_df, use_container_width=True, hide_index=True,
                column_config={
                    "Opening Stock": st.column_config.NumberColumn("Opening Stock (Yesterday)"),
                    "Arrived Morning": st.column_config.NumberColumn("☀️ Arrived Morning"),
                    "Updated Opening Stock": st.column_config.NumberColumn("🔄 Updated Opening Stock"),
                    "Unfulfilled Backlog": st.column_config.NumberColumn("🚨 Active Backlog"),
                    "Order Placed Evening": st.column_config.NumberColumn("🌙 Ordered Evening"),
                    "Total Pipeline Inventory": st.column_config.NumberColumn("📦 Total Pipeline Inventory"),
                    "Closing Inventory": st.column_config.NumberColumn("Closing Stock")
                }
            )
    else:
        st.info("💡 Interaction Required: Execute steps using the gameplay action controls above to populate tables and performance metrics.")










with tab5:
    st.header("⚖️ Cost Optimization Engine")
    st.markdown(
        "Compare your actual procurement and holding costs against mathematically optimized inventory models "
        "to identify potential profit leakage."
    )
    
    # --- STEP 1: INPUT PARAMETERS ---
    st.subheader("1. Parameters & Cost Drivers")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        holding_fixed = st.number_input("Fixed Holding Cost ($/year)", min_value=0.0, value=500.0, step=50.0, help="Warehouse insurance, security, climate control fixes.")
        holding_var_pct = st.number_input("Variable Holding Cost (% of Item Cost/year)", min_value=0.0, max_value=100.0, value=15.0, step=1.0, help="Opportunity cost of capital, damage, obsolescence.") / 100.0
        
    with col2:
        ordering_cost = st.number_input("Ordering Cost ($/order)", min_value=0.1, value=75.0, step=5.0, help="Freight flat fees, customs clearance, QA inspection, admin time.")
        lead_time_days = st.number_input("Lead Time (Days)", min_value=1, value=14, step=1)
        
    with col3:
        service_level = st.slider("Target Service Level (%)", min_value=50.0, max_value=99.9, value=95.0, step=0.5) / 100.0
        review_system = st.radio("Inventory Review System Strategy", ["Continuous Review (Q, R)", "Periodic Review (P, T)"])

    # --- STEP 2: DATA INGESTION (CSV & EXCEL SUPPORTER) ---
    st.subheader("2. Upload Historical Invoices & Demand Data")
    st.markdown("Upload a CSV or Excel file containing daily or weekly demand records alongside actual purchase orders to baseline performance.")
    
    uploaded_file = st.file_uploader(
        "Upload Inventory Ledger (CSV or Excel)", 
        type=["csv", "xlsx", "xls"], 
        help="Expected columns: 'Date', 'Demand_Qty', 'Purchase_Qty', 'Unit_Cost'"
    )
    
    # Process file based on extension types with defensive checks
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            required_cols = ["Date", "Demand_Qty", "Purchase_Qty", "Unit_Cost"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}. Please check your file formatting.")
                st.stop()
                
        except Exception as e:
            st.error(f"Error parsing file: {e}")
            st.stop()
            
    else:
        st.info("💡 Using a simulated 365-day ledger. Upload your own data above to customize the audit.")
        np.random.seed(42)
        dates = pd.date_range(start="2025-01-01", periods=365)
        demand = np.random.normal(loc=50, scale=12, size=365).clip(0).astype(int)
        
        purchase = np.zeros(365)
        purchase_indices = [15, 60, 110, 160, 210, 260, 315]
        for idx in purchase_indices:
            purchase[idx] = 2600
            
        df = pd.DataFrame({
            "Date": dates,
            "Demand_Qty": demand,
            "Purchase_Qty": purchase,
            "Unit_Cost": [25.0] * 365
        })

    # --- STEP 3: STATISTICAL ANALYSIS & CORES ---
    total_demand = df["Demand_Qty"].sum()
    avg_daily_demand = df["Demand_Qty"].mean()
    std_daily_demand = df["Demand_Qty"].std()
    avg_unit_cost = df["Unit_Cost"].mean()
    
    annual_demand = avg_daily_demand * 365
    unit_holding_cost = (holding_fixed / max(1, total_demand)) + (avg_unit_cost * holding_var_pct)
    
    lt_demand_mean = avg_daily_demand * lead_time_days
    lt_demand_std = std_daily_demand * np.sqrt(lead_time_days)
    
    z_val = stats.norm.ppf(service_level)
    
    # --- STEP 4: MODEL RECOMMENDATIONS & CALCULATIONS ---
    st.subheader("3. Optimization Recommendations")
    
    # Calculate Actuals from Ledger Data
    actual_orders_placed = np.count_nonzero(df["Purchase_Qty"])
    actual_total_ordering_cost = actual_orders_placed * ordering_cost
    
    # Trace inventory behavior (using stable 1.25x trigger logic)
    current_inv = 1.25 * lt_demand_mean  
    inv_levels = []
    for _, row in df.iterrows():
        current_inv += row["Purchase_Qty"] - row["Demand_Qty"]
        inv_levels.append(max(0, current_inv))
    
    actual_avg_inventory = np.mean(inv_levels)
    actual_total_holding_cost = actual_avg_inventory * unit_holding_cost
    actual_total_cost = actual_total_ordering_cost + actual_total_holding_cost

    # Calculate Optimal Models
    if review_system == "Continuous Review (Q, R)":
        optimal_q = np.sqrt((2 * annual_demand * ordering_cost) / unit_holding_cost)
        safety_stock = z_val * lt_demand_std
        reorder_point = lt_demand_mean + safety_stock
        
        optimal_ordering_cost = (annual_demand / optimal_q) * ordering_cost
        optimal_holding_cost = ((optimal_q / 2) + safety_stock) * unit_holding_cost
        optimal_total_cost = optimal_ordering_cost + optimal_holding_cost
        
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        rec_col1.metric("Recommended Order Quantity (Q)", f"{int(optimal_q)} units")
        rec_col2.metric("Reorder Point (ROP)", f"{int(reorder_point)} units")
        rec_col3.metric("Safety Stock Allocated", f"{int(safety_stock)} units")
        
    else:  # Periodic Review (P, T)
        optimal_p_years = np.sqrt((2 * ordering_cost) / (unit_holding_cost * annual_demand))
        optimal_p_days = max(1, int(optimal_p_years * 365))
        
        total_time_horizon = optimal_p_days + lead_time_days
        p_lt_demand_mean = avg_daily_demand * total_time_horizon
        p_lt_demand_std = std_daily_demand * np.sqrt(total_time_horizon)
        
        safety_stock = z_val * p_lt_demand_std
        order_up_to = p_lt_demand_mean + safety_stock
        
        # Approximate equivalent Q for cost structures
        optimal_q = avg_daily_demand * optimal_p_days
        optimal_ordering_cost = (365 / optimal_p_days) * ordering_cost
        optimal_holding_cost = ((optimal_q / 2) + safety_stock) * unit_holding_cost
        optimal_total_cost = optimal_ordering_cost + optimal_holding_cost
        
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        rec_col1.metric("Optimal Review Period (P)", f"{optimal_p_days} Days")
        rec_col2.metric("Order Up-To Level (T)", f"{int(order_up_to)} units")
        rec_col3.metric("Safety Stock Allocated", f"{int(safety_stock)} units")

    # --- STEP 5: COST COMPARISON & VISUALIZATION ---
    st.subheader("4. Cost Comparison: Actual vs. Optimized Model")
    
    leakage = actual_total_cost - optimal_total_cost
    
    if leakage > 0:
        st.error(f"⚠️ **Annual Profit Leakage Detected:** ${leakage:,.2f} could be saved by optimizing order policies.")
    else:
        st.success("🎉 Your historical procurement pattern matches or outperforms the theoretical model balance!")

    # Dynamic metrics building for table view
    expected_orders = (annual_demand / optimal_q) if review_system == "Continuous Review (Q, R)" else (365 / optimal_p_days)
    expected_avg_stock = ((optimal_q / 2) + safety_stock)

    comparison_data = {
        "Metric & Operational Drivers": [
            "Average Inventory Level (Units)",
            "Total Orders Placed (Per Year)",
            "Annual Ordering Cost ($)",
            "Annual Holding Cost ($)",
            "Total Operational Cost ($)"
        ],
        "Actual Historical": [
            f"{int(actual_avg_inventory):,}",
            f"{actual_orders_placed}",
            f"${actual_total_ordering_cost:,.2f}",
            f"${actual_total_holding_cost:,.2f}",
            f"${actual_total_cost:,.2f}"
        ],
        f"Optimized ({review_system.split(' ')[0]})": [
            f"{int(expected_avg_stock):,}",
            f"{expected_orders:.1f}",
            f"${optimal_ordering_cost:,.2f}",
            f"${optimal_holding_cost:,.2f}",
            f"${optimal_total_cost:,.2f}"
        ],
        "Variance / Potential Savings": [
            f"{int(actual_avg_inventory - expected_avg_stock):,}",
            f"{actual_orders_placed - expected_orders:+.1f}",
            f"${actual_total_ordering_cost - optimal_ordering_cost:,.2f}",
            f"${actual_total_holding_cost - optimal_holding_cost:,.2f}",
            f"${leakage:,.2f}"
        ]
    }
    
    comp_df = pd.DataFrame(comparison_data)
    
    st.markdown("**Financial & Operational Summary Table**")
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    st.caption("Note: Variance figures show excess units/spend. Positive financial values indicate direct cost-reduction opportunities.")

    # High-contrast corporate blue-themed comparison bar layout
    categories = ['Ordering Cost', 'Holding Cost', 'Total Cost']
    
    fig = go.Figure(data=[
        go.Bar(
            name='Actual Historical Cost', 
            x=categories, 
            y=[actual_total_ordering_cost, actual_total_holding_cost, actual_total_cost],
            marker_color='#B0C4DE'
        ),
        go.Bar(
            name='Optimized Policy Cost', 
            x=categories, 
            y=[optimal_ordering_cost, optimal_holding_cost, optimal_total_cost],
            marker_color='#1F77B4'
        )
    ])
    
    fig.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_title="USD ($)",
        font=dict(color="#333333"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=20, b=40)
    )
    
    fig.update_yaxes(showgrid=True, gridcolor='#E5E5E5')
    st.plotly_chart(fig, use_container_width=True)
