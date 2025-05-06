import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set konfigurasi halaman
st.set_page_config(
    page_title="Statistik dan History Training Model", page_icon="🧠", layout="wide"
)

# CSS untuk styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 0 5px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #0068c9;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f2ff;
        border-bottom: 2px solid #0068c9;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def safe_format(value, format_style=None, decimal_places=4, thousands_sep=False):
    """
    Safely format a value with appropriate formatting based on the value type or specified format style.
    
    Args:
        value: The value to format
        format_style: Optional style specification ('percentage', 'r2', 'time', 'integer')
        decimal_places: Number of decimal places for float values
        thousands_sep: Whether to include thousands separators
        
    Returns:
        Formatted string representation of the value
    """
    # Handle None or NaN values
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
        
    # Apply format based on style if specified
    if format_style:
        if format_style == 'percentage':
            # Format as percentage with 2 decimal places
            if isinstance(value, (int, float)):
                return f"{value:.2f}"
        elif format_style == 'r2':
            # Format R² with 4 decimal places
            if isinstance(value, (int, float)):
                return f"{value:.4f}"
        elif format_style == 'time':
            # Format time with 2 decimal places
            if isinstance(value, (int, float)):
                return f"{value:.2f}"
        elif format_style == 'integer':
            # Always use thousands separator for integers
            if isinstance(value, int):
                return f"{value:,}"
    
    # Default formatting based on type
    if isinstance(value, int):
        if thousands_sep:
            return f"{value:,}"
        return str(value)
    if isinstance(value, float):
        format_str = f"{{:.{decimal_places}f}}"
        return format_str.format(value)
    return str(value)

def load_model_metadata():
    """Load model metadata from JSON file"""
    metadata_path = "model/optimized_rf_model_metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None

def load_model_params():
    """Load detailed model parameters from text file"""
    params_path = "model/optimized_rf_model_params.txt"
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            content = f.read()
        return content
    return None

def load_model():
    """Load the trained model"""
    model_path = "model/optimized_rf_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

def check_evaluation_files():
    """Check if evaluation files exist and return their paths"""
    evaluation_dir = "model/evaluation"
    files = {
        "feature_importance": None,
        "model_comparison": None,
        "prediction_scatter": None,
        "residual_plot": None
    }
    
    if os.path.exists(os.path.join(evaluation_dir, "feature_importance.png")):
        files["feature_importance"] = os.path.join(evaluation_dir, "feature_importance.png")
    
    if os.path.exists(os.path.join(evaluation_dir, "rf_model_comparison.png")):
        files["model_comparison"] = os.path.join(evaluation_dir, "rf_model_comparison.png")
    
    if os.path.exists(os.path.join(evaluation_dir, "prediction_scatter.png")):
        files["prediction_scatter"] = os.path.join(evaluation_dir, "prediction_scatter.png")
    
    if os.path.exists(os.path.join(evaluation_dir, "residual_plot.png")):
        files["residual_plot"] = os.path.join(evaluation_dir, "residual_plot.png")
    
    return files

def display_metric_card(title, value, unit="", help_text=""):
    """Display a metric in a styled card"""
    value_display = value
    if unit and value != "N/A" and not isinstance(value, str):
        value_display = f"{value} {unit}"
    elif unit and value != "N/A":
        value_display = f"{value}{unit}"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value_display}</div>
        <div class="metric-label" style="font-size: 12px; margin-top: 5px;">{help_text}</div>
    </div>
    """, unsafe_allow_html=True)

# Main interface
st.title("🧠 Statistik dan History Training Model")
st.markdown(
    """
    Halaman ini menampilkan statistik dan history training model yang digunakan untuk memprediksi harga rumah.
    Model yang digunakan adalah Random Forest Regressor. Model ini dilatih menggunakan data yang telah dibersihkan dan diproses sebelumnya.
    """
)

# Check if model exists
model_exists = os.path.exists("model/optimized_rf_model.pkl")

if not model_exists:
    st.warning("⚠️ Model belum dilatih. Silakan latih model terlebih dahulu di halaman Training Model.")
    st.stop()

# Load model metadata and other resources
with st.spinner("Memuat data model..."):
    metadata = load_model_metadata()
    model_params = load_model_params()
    evaluation_files = check_evaluation_files()
    model = load_model()

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Overview", 
    "📈 Performance Metrics", 
    "🔍 Feature Analysis", 
    "🔄 Model Comparison"
])

# Tab 1: Model Overview
with tab1:
    st.header("Model Overview")
    
    if metadata:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            display_metric_card(
                "MAPE Score", 
                safe_format(metadata.get('mape', 'N/A'), format_style='percentage'), 
                unit="%",
                help_text="Mean Absolute Percentage Error - lower is better"
            )
        
        with col2:
            display_metric_card(
                "Training Date", 
                metadata.get('training_date', 'Unknown'), 
                help_text="When the model was last trained"
            )
        
        with col3:
            display_metric_card(
                "Data Points", 
                safe_format(metadata.get('data_points', 'N/A'), format_style='integer'), 
                help_text="Number of samples used for training"
            )
        
        st.subheader("Model Details")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Key Parameters")
            params = metadata.get('parameters', {})
            for param, value in params.items():
                st.write(f"**{param}:** {value}")
        
        with col2:
            st.markdown("### Training Metrics")
            metrics = metadata.get('metrics', {})
            if metrics:
                for metric, value in metrics.items():
                    if metric != 'mape':  # MAPE already displayed above
                        format_style = None
                        # Set format style based on metric name
                        if 'r2' in metric.lower() or 'score' in metric.lower():
                            format_style = 'r2'
                        elif 'time' in metric.lower():
                            format_style = 'time'
                        elif 'percentage' in metric.lower() or 'error' in metric.lower():
                            format_style = 'percentage'
                        # Format and display the metric
                        st.write(f"**{metric}:** {safe_format(value, format_style=format_style)}")
            
            # Time metrics
            time_metrics = metadata.get('training_time', {})
            if time_metrics:
                st.markdown("### Training Time")
                st.write(f"**Total time:** {safe_format(time_metrics.get('total', 'N/A'), format_style='time')} seconds")
                st.write(f"**Optimization time:** {safe_format(time_metrics.get('optimization', 'N/A'), format_style='time')} seconds")
        
        if model_params:
            with st.expander("Detailed Model Parameters"):
                st.code(model_params, language="text")
    else:
        st.error("Model metadata not found. The model may not have been properly trained.")

# Tab 2: Performance Metrics
with tab2:
    st.header("Performance Metrics")
    
    if metadata:
        # Performance metrics visualization
        metrics = metadata.get('metrics', {})
        if metrics:
            # Create a bar chart for metrics comparison
            metric_df = pd.DataFrame({
                'Metric': list(metrics.keys()),
                'Value': list(metrics.values())
            })
            
            fig = px.bar(
                metric_df, 
                x='Metric', 
                y='Value', 
                title='Model Performance Metrics',
                color='Value',
                color_continuous_scale='blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Display prediction scatter plot if available
        if evaluation_files["prediction_scatter"]:
            st.subheader("Actual vs Predicted Values")
            st.image(evaluation_files["prediction_scatter"], use_column_width=True)
        
        # Display residual plot if available
        if evaluation_files["residual_plot"]:
            st.subheader("Residual Plot")
            st.image(evaluation_files["residual_plot"], use_column_width=True)
            
        # If we don't have those plots but have the model, we could generate them
        if model and (not evaluation_files["prediction_scatter"] or not evaluation_files["residual_plot"]):
            st.info("Some visualization plots are missing. You may need to regenerate them by evaluating the model.")
    else:
        st.error("Model performance metrics not found. The model may not have been properly evaluated.")

# Tab 3: Feature Analysis
with tab3:
    st.header("Feature Analysis")
    
    if evaluation_files["feature_importance"]:
        st.subheader("Feature Importance")
        st.image(evaluation_files["feature_importance"], use_column_width=True)
        
        # If we have the model, we can display more detailed feature importance
        if model and hasattr(model, 'feature_importances_'):
            try:
                # Load the feature names from metadata if available
                feature_names = metadata.get('feature_names', [f"Feature {i}" for i in range(len(model.feature_importances_))])
                
                # Create a DataFrame for the feature importances
                fi_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Display as a table
                st.subheader("Feature Importance Values")
                st.dataframe(fi_df, use_container_width=True)
                
                # Interactive bar chart
                fig = px.bar(
                    fi_df.head(15), 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title='Top 15 Most Important Features',
                    color='Importance',
                    color_continuous_scale='blues'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating feature importance plot: {str(e)}")
    else:
        st.warning("Feature importance visualization not found. You may need to regenerate it by evaluating the model.")

# Tab 4: Model Comparison
with tab4:
    st.header("Model Comparison")
    
    if evaluation_files["model_comparison"]:
        st.subheader("Base vs Optimized Model Performance")
        st.image(evaluation_files["model_comparison"], use_column_width=True)
    
    # If we have comparison data in the metadata
    if metadata and 'comparison' in metadata:
        comparison = metadata['comparison']
        
        # Create comparison dataframe
        models = []
        mapes = []
        r2s = []
        
        if 'base_model' in comparison:
            models.append('Base Model')
            mapes.append(comparison['base_model'].get('mape', 0))
            r2s.append(comparison['base_model'].get('r2', 0))
        
        if 'optimized_model' in comparison:
            models.append('Optimized Model')
            mapes.append(comparison['optimized_model'].get('mape', 0))
            r2s.append(comparison['optimized_model'].get('r2', 0))
        
        if models:
            comp_df = pd.DataFrame({
                'Model': models,
                'MAPE': mapes,
                'R² Score': r2s
            })
            
            # Calculate improvement percentage
            if len(models) > 1:
                improvement = ((mapes[0] - mapes[1]) / mapes[0]) * 100 if mapes[0] != 0 else 0
                st.metric(
                    label="Improvement in MAPE", 
                    value=f"{safe_format(improvement, format_style='percentage')}%", 
                    delta=f"{safe_format(improvement, format_style='percentage')}%"
                )
            
            # Display comparison chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=comp_df['Model'],
                y=comp_df['MAPE'],
                name='MAPE (lower is better)',
                marker_color='indianred'
            ))
            
            fig.add_trace(go.Bar(
                x=comp_df['Model'],
                y=comp_df['R² Score'],
                name='R² Score (higher is better)',
                marker_color='lightsalmon'
            ))
            
            fig.update_layout(
                title_text='Model Performance Comparison',
                xaxis_title='Model',
                yaxis_title='Score',
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display as table as well
            st.dataframe(comp_df, use_container_width=True)
    else:
        st.warning("Model comparison data not found. You may need to compare the models by evaluating them.")

# Sidebar with additional information and actions
with st.sidebar:
    st.header("Model Information")
    
    if metadata:
        st.info(f"""
        **Model Version**: {metadata.get('version', 'v1.0')}
        **Model Type**: Random Forest Regressor
        **Training Date**: {metadata.get('training_date', 'Unknown')}
        **MAPE Score**: {safe_format(metadata.get('mape', 'N/A'), format_style='percentage')}%
        """)
        
        if 'data_split' in metadata:
            split = metadata['data_split']
            st.markdown("### Data Split")
            st.write(f"**Training**: {safe_format(split.get('train', 'N/A'), format_style='integer')}")
            st.write(f"**Validation**: {safe_format(split.get('val', 'N/A'), format_style='integer')}")
            st.write(f"**Test**: {safe_format(split.get('test', 'N/A'), format_style='integer')}")
    else:
        st.warning("No model metadata available")
    
    st.markdown("---")
    st.markdown("### Actions")
    
    # Button to refresh model stats if model was retrained
    if st.button("🔄 Refresh Model Statistics"):
        st.experimental_rerun()
        
    st.caption("This page displays statistics and results from the trained model. To train or retrain models, use the training functionality in your notebook or scripts.")
