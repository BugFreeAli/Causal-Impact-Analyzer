# app.py
import streamlit as st
import pandas as pd
import numpy as np
from causalimpact import CausalImpact
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# Page Configuration & Setup
# ----------------------------
st.set_page_config(
    page_title="Causal Impact Analyzer | Professional",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for persistence across reruns
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'ci' not in st.session_state:
    st.session_state.ci = None
if 'analysis_df' not in st.session_state:
    st.session_state.analysis_df = None
if 'event_date' not in st.session_state:
    st.session_state.event_date = None

# ----------------------------
# Custom CSS for a Professional UI
# ----------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .subheader {
        font-size: 1.2rem;
        color: #7f7f7f;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: 600;
    }
    .result-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        border-radius: 4px;
        padding: 1.5rem;
        margin: 1rem 0rem;
    }
    .metric-box {
        background-color: #e8f4f8;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# App Header
# ----------------------------
st.markdown('<p class="main-header">🔍 Causal Impact Analyzer</p>', unsafe_allow_html=True)
st.markdown("""
<p class="subheader">A professional tool for measuring the true causal effect of interventions on time series data using Bayesian Structural Time Series.</p>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar - User Inputs & Configuration
# ----------------------------
st.sidebar.header("📁 Data Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload your time series data (CSV)",
    type=['csv'],
    help="Ensure your CSV has a date column and at least one numeric metric column."
)

# Advanced options in an expander
with st.sidebar.expander("⚙️ Advanced Analysis Options"):
    st.markdown("**Model Parameters**")
    nseasons = st.slider("Seasonality Period", min_value=1, max_value=365, value=7, help="Number of data points in each season (e.g., 7 for weekly seasonality).")
    model_args = st.text_area("Custom Model Args", value="", help="Advanced: Add custom `model_args` for CausalImpact as a Python dictionary (e.g., {'niter': 1000}). Use with caution.")

# If a file is uploaded, try to load it and get column names
if uploaded_file is not None:
    try:
        # Try reading the file
        df = pd.read_csv(uploaded_file)
        col_options = df.columns.tolist()

        st.sidebar.success("✅ File successfully uploaded!")

        # Let user choose the date and metric column
        date_col = st.sidebar.selectbox(
            "Select the DATE column",
            options=col_options,
            index=0,
            help="This column should contain the dates for your time series."
        )
        metric_col = st.sidebar.selectbox(
            "Select the METRIC column",
            options=col_options,
            index=1 if len(col_options) > 1 else 0,
            help="This column should contain the numeric values you want to analyze (e.g., Sales, Users)."
        )

        # Convert the date column to datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        if df[date_col].isnull().any():
            st.sidebar.error("❌ Error: Could not convert some dates. Please check your date column format.")
            st.stop()

        # Let user choose the event date from the available dates
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        # Default to 70% through the time period for a reasonable pre-period
        default_event_date = min_date + (max_date - min_date) * 0.7

        event_date = st.sidebar.date_input(
            "Select the EVENT date",
            value=default_event_date,
            min_value=min_date,
            max_value=max_date
        )
        event_date = pd.to_datetime(event_date)
        st.session_state.event_date = event_date

        # Set the date column as the index
        analysis_df = df.set_index(date_col)[[metric_col]].sort_index()

        # --- ENHANCED DATA VALIDATION AND CLEANING ---
        # 1. Check for and handle duplicate dates
        if analysis_df.index.duplicated().any():
            dup_count = analysis_df.index.duplicated().sum()
            st.sidebar.warning(f"⚠️ Found {dup_count} duplicate date(s). Aggregating by mean.")
            analysis_df = analysis_df.groupby(analysis_df.index).mean()

        # 2. Ensure the index is a proper DateTimeIndex
        analysis_df.index = pd.DatetimeIndex(analysis_df.index)

        # 3. Create a complete date range to identify missing days
        full_date_range = pd.date_range(
            start=analysis_df.index.min(),
            end=analysis_df.index.max(),
            freq='D'
        )

        # 4. Reindex to the complete date range, introducing NaNs for missing days
        analysis_df = analysis_df.reindex(full_date_range)

        # 5. Handle missing values: Offer user a choice
        missing_days = analysis_df[metric_col].isnull().sum()
        if missing_days > 0:
            st.sidebar.info(f"📅 Found {missing_days} missing days.")
            handle_missing = st.sidebar.radio(
                "Handle missing values by:",
                options=['Forward Fill', 'Linear Interpolation', 'Leave as NaN (Not Recommended)'],
                index=0
            )
            if handle_missing == 'Forward Fill':
                analysis_df[metric_col] = analysis_df[metric_col].fillna(method='ffill')
            elif handle_missing == 'Linear Interpolation':
                analysis_df[metric_col] = analysis_df[metric_col].interpolate(method='linear')

        st.session_state.analysis_df = analysis_df

        # --- DATA SUMMARY STATISTICS ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Data Summary")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Start Date", analysis_df.index.min().strftime('%Y-%m-%d'))
        with col2:
            st.metric("End Date", analysis_df.index.max().strftime('%Y-%m-%d'))
        col3, col4 = st.sidebar.columns(2)
        with col3:
            st.metric("Total Days", len(analysis_df))
        with col4:
            non_missing = len(analysis_df) - analysis_df[metric_col].isnull().sum()
            st.metric("Days with Data", f"{non_missing}")

    except Exception as e:
        st.sidebar.error(f"❌ Error reading file: {str(e)}")
        st.stop()
else:
    # Show instructions if no file is uploaded
    st.info(
        """
        👋 **Welcome to the Causal Impact Analyzer!**

        **To get started:**
        1.  **Upload a CSV file** using the sidebar on the left.
        2.  Ensure your file has at least two columns:
            - A **date column** (e.g., 'Date', 'Day')
            - A **numeric metric column** (e.g., 'Sales', 'Users', 'Visits')
        3.  Select the appropriate columns and the date of the event you want to analyze.
        4.  Configure any advanced options if needed.
        5.  Click **'Run Causal Impact Analysis'** to see the results!

        **For best results:** Ensure you have sufficient data before the event (pre-period) to establish a reliable baseline trend.
        """
    )
    # Example DataFrame display
    st.subheader("📋 Example of Expected Data Format")
    example_data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'Sales': np.random.randn(100).cumsum() + 100  # Synthetic trending data
    })
    example_data.loc[70:, 'Sales'] += 15  # Add an intervention effect
    st.dataframe(example_data.head(10), use_container_width=True)
    st.caption("Example data with 100 days of synthetic 'Sales' data and an intervention effect starting at day 70.")

    st.stop()

# ----------------------------
# Main Panel Logic
# ----------------------------
tab1, tab2, tab3 = st.tabs(["📈 Data Overview", "🔍 Run Analysis", "📤 Export Results"])

with tab1:
    st.subheader("Data Overview & Preprocessing")

    st.dataframe(st.session_state.analysis_df.head(10), use_container_width=True)
    st.caption(f"Preview of your processed data (showing 10 of {len(st.session_state.analysis_df)} rows).")

    # Create a plot of the original data
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(
        x=st.session_state.analysis_df.index,
        y=st.session_state.analysis_df[metric_col],
        mode='lines+markers',
        name=metric_col,
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4)
    ))
    # Add event line using shape
    fig_raw.add_shape(
        type="line",
        x0=st.session_state.event_date,
        y0=0,
        x1=st.session_state.event_date,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="red", width=3, dash="dash")
    )
    fig_raw.add_annotation(
        x=st.session_state.event_date,
        y=1,
        xref="x",
        yref="paper",
        text="Event",
        showarrow=False,
        yanchor="bottom",
        xshift=10,
        font=dict(color="red", size=12)
    )
    fig_raw.update_layout(
        title=f'Raw Data with Event Marker: {metric_col}',
        xaxis_title='Date',
        yaxis_title=metric_col,
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig_raw, use_container_width=True)

    # Data summary statistics
    st.subheader("Data Quality Report")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{st.session_state.analysis_df[metric_col].mean():.2f}")
    with col2:
        st.metric("Std Dev", f"{st.session_state.analysis_df[metric_col].std():.2f}")
    with col3:
        st.metric("Min", f"{st.session_state.analysis_df[metric_col].min():.2f}")
    with col4:
        st.metric("Max", f"{st.session_state.analysis_df[metric_col].max():.2f}")

with tab2:
    st.subheader("Run Causal Impact Analysis")

    if st.button("🚀 Run Causal Impact Analysis", type="primary", use_container_width=True):
        st.session_state.analysis_complete = False

        with st.spinner('Running advanced Bayesian analysis... This may take a moment for large datasets.'):
            try:
                # Define pre and post periods
                pre_period = [st.session_state.analysis_df.index.min(), st.session_state.event_date - timedelta(days=1)]
                post_period = [st.session_state.event_date, st.session_state.analysis_df.index.max()]

                # Prepare model args
                custom_model_args = {}
                if model_args.strip():
                    try:
                        custom_model_args = eval(model_args)
                        if not isinstance(custom_model_args, dict):
                            st.warning("Custom model args must be a dictionary. Using defaults.")
                            custom_model_args = {}
                    except:
                        st.warning("Could not parse custom model args. Using defaults.")
                        custom_model_args = {}

                # Default model args
                default_model_args = {'nseasons': nseasons, 'season_duration': 1}
                # Merge defaults with custom args (custom args take precedence)
                final_model_args = {**default_model_args, **custom_model_args}

                # Run the CausalImpact model
                ci = CausalImpact(st.session_state.analysis_df, pre_period, post_period, model_args=final_model_args)

                st.session_state.ci = ci
                st.session_state.analysis_complete = True

                st.success("✅ Analysis completed successfully!")

            except Exception as e:
                st.error(f"❌ An error occurred during analysis: {str(e)}")
                st.info("""
                **Common causes:**
                - Pre-event period is too short to build a reliable model
                - Data is too volatile or has insufficient signal
                - Custom model arguments are malformed
                - Try increasing the pre-period duration or checking your data quality.
                """)

        if st.session_state.analysis_complete and st.session_state.ci is not None:
        # Display the summary in a nice box
            st.markdown("### 📋 Analysis Summary Report")
            st.markdown('<div class="result-box">', unsafe_allow_html=True)

            # Get the summary as text and display it
            summary_output = st.session_state.ci.summary(output='report')
            st.text(summary_output)

            # Extract and display key metrics
            st.markdown("### 📊 Key Impact Metrics")
            
            # Very simple approach - just show the summary and p-value
            summary_text = st.session_state.ci.summary(output='report')
            
            # Get p-value if available
            p_value = getattr(st.session_state.ci, 'p_value', 0.5)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Analysis Complete", "✅ Success")
            with col2:
                st.metric("Data Points", len(st.session_state.analysis_df))
            with col3:
                sig_status = "✅ Significant" if p_value <= 0.05 else "❌ Not Significant"
                st.metric("Statistical Significance", sig_status, f"p={p_value:.3f}")
            
            # Show the full summary
            st.markdown("#### 📋 Full Summary Report")
            st.text(summary_text)
            
            st.markdown('</div>', unsafe_allow_html=True)

            # --- Create Interactive Plotly Chart from results ---
            st.markdown("### 📈 Detailed Analysis Plot")
            impact_data = st.session_state.ci.inferences.copy()
            impact_data.index = st.session_state.analysis_df.index

            # DEBUG: Let's see the actual data structure
            st.write("Actual columns in inferences data:", impact_data.columns.tolist())
            
            # AUTOMATICALLY DETECT THE CORRECT COLUMN NAMES
            actual_col = None
            pred_col = None
            upper_col = None
            lower_col = None
            
            # Look for columns that likely contain the actual data
            for col in impact_data.columns:
                col_lower = str(col).lower()
                # Look for actual data columns
                if 'actual' in col_lower or 'response' in col_lower or 'y' in col_lower or 'observed' in col_lower:
                    actual_col = col
                # Look for prediction columns
                elif 'pred' in col_lower and 'upper' not in col_lower and 'lower' not in col_lower:
                    pred_col = col
                # Look for confidence interval columns
                elif 'upper' in col_lower:
                    upper_col = col
                elif 'lower' in col_lower:
                    lower_col = col
            
            # If we couldn't find specific names, use the first columns
            if actual_col is None and len(impact_data.columns) > 0:
                actual_col = impact_data.columns[0]
            if pred_col is None and len(impact_data.columns) > 1:
                pred_col = impact_data.columns[1]
            if upper_col is None and len(impact_data.columns) > 2:
                upper_col = impact_data.columns[2]
            if lower_col is None and len(impact_data.columns) > 3:
                lower_col = impact_data.columns[3]

            st.write(f"Detected columns: actual={actual_col}, predicted={pred_col}, upper={upper_col}, lower={lower_col}")

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=('Actual vs. Predicted', 'Pointwise Effect'))

            # Plot 1: Actual vs Predicted
            fig.add_trace(go.Scatter(
                x=impact_data.index, 
                y=impact_data[actual_col],
                mode='lines', 
                name='Actual', 
                line=dict(color='blue')
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=impact_data.index, 
                y=impact_data[pred_col],
                mode='lines', 
                name='Predicted', 
                line=dict(color='orange', dash='dash')
            ), row=1, col=1)

            # Add confidence interval if available
            if upper_col and lower_col and upper_col in impact_data.columns and lower_col in impact_data.columns:
                fig.add_trace(go.Scatter(
                    x=impact_data.index, 
                    y=impact_data[upper_col],
                    mode='lines', 
                    line=dict(width=0), 
                    showlegend=False
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=impact_data.index, 
                    y=impact_data[lower_col],
                    mode='lines', 
                    line=dict(width=0), 
                    fill='tonexty',
                    fillcolor='rgba(255, 165, 0, 0.2)', 
                    name='95% CI'
                ), row=1, col=1)
            else:
                st.warning("Confidence interval columns not found")

            # Plot 2: Pointwise Effect
            pointwise_effect = impact_data[actual_col] - impact_data[pred_col]
            fig.add_trace(go.Scatter(
                x=impact_data.index, 
                y=pointwise_effect,
                mode='lines', 
                name='Pointwise Effect', 
                line=dict(color='green')
            ), row=2, col=1)

            # Add the event line to both subplots
            for row in [1, 2]:
                fig.add_shape(
                    type="line",
                    x0=st.session_state.event_date,
                    y0=0,
                    x1=st.session_state.event_date,
                    y1=1,
                    xref="x",
                    yref="paper",
                    row=row,
                    col=1,
                    line=dict(color="red", width=2, dash="dash")
                )

            fig.update_layout(
                height=700,
                title_text="Causal Impact Analysis",
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_yaxes(title_text=metric_col, row=1, col=1)
            fig.update_yaxes(title_text="Effect Size", row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)

            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Export Results")

    if st.session_state.analysis_complete and st.session_state.ci is not None:
        # Export Summary as Text
        st.download_button(
            label="📄 Download Summary Report",
            data=st.session_state.ci.summary(output='report'),
            file_name="causal_impact_summary.txt",
            mime="text/plain",
            use_container_width=True
        )

        # Export Chart as PNG
        # (Note: This requires kaleido - we'll add it to requirements if needed)
        try:
            img_bytes = fig.to_image(format="png", scale=2)
            st.download_button(
                label="🖼️ Download Chart as PNG",
                data=img_bytes,
                file_name="causal_impact_chart.png",
                mime="image/png",
                use_container_width=True
            )
        except:
            st.info("Install `kaleido` to enable PNG chart downloads.")

        # Export Data as CSV
        impact_data = st.session_state.ci.inferences.copy()
        impact_data.index = st.session_state.analysis_df.index
                # Try to get the actual data - handle different versions
                # Add the actual data to export
        try:
            impact_data['actual'] = st.session_state.analysis_df[metric_col]
        except:
            st.warning("Could not add actual data to export")
        csv = impact_data.to_csv()
        st.download_button(
            label="📊 Download Results Data (CSV)",
            data=csv,
            file_name="causal_impact_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Run an analysis first to enable export options.")