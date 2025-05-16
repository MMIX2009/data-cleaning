import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Data Cleaning Assistant", layout="wide")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'cleaning_log' not in st.session_state:
    st.session_state.cleaning_log = []

# Helper functions
def log_cleaning_step(step_name, code, reasoning, before_shape, after_shape):
    """Log a cleaning step with code, reasoning, and shape changes"""
    st.session_state.cleaning_log.append({
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'step': step_name,
        'code': code,
        'reasoning': reasoning,
        'before_shape': before_shape,
        'after_shape': after_shape
    })

def display_cleaning_log():
    """Display the cleaning log with expandable sections"""
    if st.session_state.cleaning_log:
        st.subheader("🔍 Data Cleaning Log")
        for i, log_entry in enumerate(st.session_state.cleaning_log):
            with st.expander(f"{log_entry['timestamp']} - {log_entry['step']} (Shape: {log_entry['before_shape']} → {log_entry['after_shape']})"):
                st.write("**Reasoning:**", log_entry['reasoning'])
                st.code(log_entry['code'], language='python')

def get_column_info(df):
    """Get detailed information about each column"""
    info = []
    for col in df.columns:
        col_info = {
            'Column': col,
            'Data Type': str(df[col].dtype),
            'Non-Null Count': df[col].count(),
            'Null Count': df[col].isnull().sum(),
            'Null Percentage': f"{(df[col].isnull().sum() / len(df) * 100):.2f}%",
            'Unique Values': df[col].nunique(),
            'Sample Values': str(df[col].dropna().head(3).tolist())
        }
        info.append(col_info)
    return pd.DataFrame(info)

def automated_data_cleaning(df):
    """Perform automated data cleaning with documentation"""
    cleaned_df = df.copy()
    cleaning_steps = []
    
    # 1. Remove completely empty rows and columns
    before_shape = cleaned_df.shape
    cleaned_df = cleaned_df.dropna(how='all')
    cleaned_df = cleaned_df.dropna(axis=1, how='all')
    after_shape = cleaned_df.shape
    if before_shape != after_shape:
        log_cleaning_step(
            "Remove Empty Rows/Columns",
            "df = df.dropna(how='all')\ndf = df.dropna(axis=1, how='all')",
            "Removed rows and columns that contain only missing values as they provide no information.",
            before_shape,
            after_shape
        )
    
    # 2. Remove duplicate rows
    before_shape = cleaned_df.shape
    cleaned_df = cleaned_df.drop_duplicates()
    after_shape = cleaned_df.shape
    if before_shape != after_shape:
        log_cleaning_step(
            "Remove Duplicates",
            "df = df.drop_duplicates()",
            f"Removed {before_shape[0] - after_shape[0]} duplicate rows to ensure data uniqueness.",
            before_shape,
            after_shape
        )
    
    # 3. Handle missing values based on data type and percentage
    for col in cleaned_df.columns:
        missing_pct = cleaned_df[col].isnull().sum() / len(cleaned_df)
        before_shape = cleaned_df.shape
        
        if missing_pct > 0.5:  # Drop columns with >50% missing
            cleaned_df = cleaned_df.drop(columns=[col])
            log_cleaning_step(
                f"Drop Column: {col}",
                f"df = df.drop(columns=['{col}'])",
                f"Column '{col}' has {missing_pct:.1%} missing values (>50%), so it was dropped.",
                before_shape,
                cleaned_df.shape
            )
        elif missing_pct > 0:
            if cleaned_df[col].dtype in ['object', 'string']:
                # For categorical data, fill with mode or 'Unknown'
                mode_val = cleaned_df[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                cleaned_df[col] = cleaned_df[col].fillna(fill_val)
                log_cleaning_step(
                    f"Fill Missing: {col}",
                    f"df['{col}'] = df['{col}'].fillna('{fill_val}')",
                    f"Filled {missing_pct:.1%} missing values in '{col}' with mode value '{fill_val}' for categorical data.",
                    before_shape,
                    cleaned_df.shape
                )
            else:
                # For numerical data, fill with median
                median_val = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(median_val)
                log_cleaning_step(
                    f"Fill Missing: {col}",
                    f"df['{col}'] = df['{col}'].fillna({median_val})",
                    f"Filled {missing_pct:.1%} missing values in '{col}' with median value {median_val} for numerical data.",
                    before_shape,
                    cleaned_df.shape
                )
    
    # 4. Basic data type optimization
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == 'object':
            # Try to convert to numeric if possible
            numeric_series = pd.to_numeric(cleaned_df[col], errors='coerce')
            if not numeric_series.isnull().all():
                before_type = str(cleaned_df[col].dtype)
                cleaned_df[col] = numeric_series
                log_cleaning_step(
                    f"Convert to Numeric: {col}",
                    f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')",
                    f"Converted '{col}' from {before_type} to numeric as it contains parseable numbers.",
                    cleaned_df.shape,
                    cleaned_df.shape
                )
    
    return cleaned_df

# Main app
st.title("🧹 Data Cleaning Assistant")
st.markdown("Upload your dataset and clean it with detailed documentation of each step!")

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file to begin data cleaning"
    )
    
    if uploaded_file is not None:
        try:
            # Load the dataset
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df.copy()
            st.session_state.original_df = df.copy()
            st.session_state.cleaning_log = []  # Reset log for new file
            
            st.success(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Display basic info
            st.subheader("Dataset Overview")
            st.write(f"**Shape:** {df.shape}")
            st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Missing values summary
            missing_summary = df.isnull().sum()
            if missing_summary.sum() > 0:
                st.subheader("Missing Values")
                missing_df = pd.DataFrame({
                    'Column': missing_summary.index,
                    'Missing Count': missing_summary.values,
                    'Missing %': (missing_summary.values / len(df) * 100).round(2)
                })
                st.dataframe(missing_df[missing_df['Missing Count'] > 0])
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

# Main content area
if st.session_state.df is not None:
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔧 Manual Cleaning", "🤖 Automated Cleaning", "📋 Cleaning Log"])
    
    with tab1:
        st.header("Data Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Shape")
            st.metric("Rows", st.session_state.df.shape[0])
            st.metric("Columns", st.session_state.df.shape[1])
            
            st.subheader("Data Types")
            dtype_counts = st.session_state.df.dtypes.value_counts()
            st.bar_chart(dtype_counts)
        
        with col2:
            st.subheader("Missing Values Heatmap")
            if st.session_state.df.isnull().sum().sum() > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(st.session_state.df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
                st.pyplot(fig)
            else:
                st.info("No missing values found in the dataset!")
        
        # Detailed column information
        st.subheader("Column Details")
        column_info = get_column_info(st.session_state.df)
        st.dataframe(column_info, use_container_width=True)
        
        # First few rows
        st.subheader("Sample Data")
        st.dataframe(st.session_state.df.head(), use_container_width=True)
        
        # Basic statistics
        st.subheader("Statistical Summary")
        st.dataframe(st.session_state.df.describe(), use_container_width=True)
    
    with tab2:
        st.header("Manual Data Cleaning")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Cleaning Options")
            
            # Remove duplicates
            if st.button("🔄 Remove Duplicates"):
                before_shape = st.session_state.df.shape
                duplicates = st.session_state.df.duplicated().sum()
                if duplicates > 0:
                    st.session_state.df = st.session_state.df.drop_duplicates()
                    log_cleaning_step(
                        "Remove Duplicates",
                        "df = df.drop_duplicates()",
                        f"Removed {duplicates} duplicate rows to ensure data uniqueness.",
                        before_shape,
                        st.session_state.df.shape
                    )
                    st.success(f"Removed {duplicates} duplicate rows!")
                else:
                    st.info("No duplicates found!")
            
            # Handle missing values
            st.subheader("Handle Missing Values")
            columns_with_missing = st.session_state.df.columns[st.session_state.df.isnull().any()].tolist()
            
            if columns_with_missing:
                selected_column = st.selectbox("Select column to handle:", columns_with_missing)
                missing_count = st.session_state.df[selected_column].isnull().sum()
                missing_pct = (missing_count / len(st.session_state.df)) * 100
                
                st.write(f"**{selected_column}** has {missing_count} missing values ({missing_pct:.2f}%)")
                
                action = st.radio(
                    "Choose action:",
                    ["Drop rows with missing values", "Fill with mean", "Fill with median", "Fill with mode", "Fill with custom value", "Drop column"]
                )
                
                if st.button("Apply Action"):
                    before_shape = st.session_state.df.shape
                    
                    if action == "Drop rows with missing values":
                        st.session_state.df = st.session_state.df.dropna(subset=[selected_column])
                        log_cleaning_step(
                            f"Drop Missing Rows: {selected_column}",
                            f"df = df.dropna(subset=['{selected_column}'])",
                            f"Dropped {missing_count} rows with missing values in '{selected_column}' to maintain data quality.",
                            before_shape,
                            st.session_state.df.shape
                        )
                    elif action == "Fill with mean" and pd.api.types.is_numeric_dtype(st.session_state.df[selected_column]):
                        mean_val = st.session_state.df[selected_column].mean()
                        st.session_state.df[selected_column].fillna(mean_val, inplace=True)
                        log_cleaning_step(
                            f"Fill with Mean: {selected_column}",
                            f"df['{selected_column}'].fillna(df['{selected_column}'].mean(), inplace=True)",
                            f"Filled missing values with mean ({mean_val:.2f}) as it's appropriate for normally distributed numerical data.",
                            before_shape,
                            st.session_state.df.shape
                        )
                    elif action == "Fill with median" and pd.api.types.is_numeric_dtype(st.session_state.df[selected_column]):
                        median_val = st.session_state.df[selected_column].median()
                        st.session_state.df[selected_column].fillna(median_val, inplace=True)
                        log_cleaning_step(
                            f"Fill with Median: {selected_column}",
                            f"df['{selected_column}'].fillna(df['{selected_column}'].median(), inplace=True)",
                            f"Filled missing values with median ({median_val}) as it's robust to outliers in numerical data.",
                            before_shape,
                            st.session_state.df.shape
                        )
                    elif action == "Fill with mode":
                        mode_val = st.session_state.df[selected_column].mode()[0] if len(st.session_state.df[selected_column].mode()) > 0 else "Unknown"
                        st.session_state.df[selected_column].fillna(mode_val, inplace=True)
                        log_cleaning_step(
                            f"Fill with Mode: {selected_column}",
                            f"df['{selected_column}'].fillna(df['{selected_column}'].mode()[0], inplace=True)",
                            f"Filled missing values with mode ({mode_val}) as it's appropriate for categorical data.",
                            before_shape,
                            st.session_state.df.shape
                        )
                    elif action == "Fill with custom value":
                        custom_value = st.text_input("Enter custom value:")
                        if custom_value:
                            st.session_state.df[selected_column].fillna(custom_value, inplace=True)
                            log_cleaning_step(
                                f"Fill with Custom: {selected_column}",
                                f"df['{selected_column}'].fillna('{custom_value}', inplace=True)",
                                f"Filled missing values with custom value ('{custom_value}') based on domain knowledge.",
                                before_shape,
                                st.session_state.df.shape
                            )
                    elif action == "Drop column":
                        st.session_state.df = st.session_state.df.drop(columns=[selected_column])
                        log_cleaning_step(
                            f"Drop Column: {selected_column}",
                            f"df = df.drop(columns=['{selected_column}'])",
                            f"Dropped column '{selected_column}' due to high missing value percentage ({missing_pct:.1f}%) making it unreliable.",
                            before_shape,
                            st.session_state.df.shape
                        )
                    
                    st.success("Action applied successfully!")
                    st.rerun()
            else:
                st.info("No missing values found in the dataset!")
            
            # Remove columns
            st.subheader("Remove Columns")
            columns_to_remove = st.multiselect("Select columns to remove:", st.session_state.df.columns)
            if columns_to_remove and st.button("Remove Selected Columns"):
                before_shape = st.session_state.df.shape
                st.session_state.df = st.session_state.df.drop(columns=columns_to_remove)
                log_cleaning_step(
                    f"Remove Columns",
                    f"df = df.drop(columns={columns_to_remove})",
                    f"Removed columns {columns_to_remove} as they are not needed for analysis.",
                    before_shape,
                    st.session_state.df.shape
                )
                st.success(f"Removed {len(columns_to_remove)} columns!")
                st.rerun()
        
        with col2:
            st.subheader("Current Dataset")
            st.dataframe(st.session_state.df.head(20), use_container_width=True)
            
            # Show current statistics
            st.subheader("Current Statistics")
            col1_stats, col2_stats = st.columns(2)
            with col1_stats:
                st.metric("Rows", st.session_state.df.shape[0])
                st.metric("Columns", st.session_state.df.shape[1])
            with col2_stats:
                st.metric("Missing Values", st.session_state.df.isnull().sum().sum())
                st.metric("Duplicates", st.session_state.df.duplicated().sum())
    
    with tab3:
        st.header("Automated Data Cleaning")
        st.markdown("""
        This automated cleaning will perform the following steps:
        1. **Remove empty rows and columns** - Drops rows/columns with all missing values
        2. **Remove duplicates** - Eliminates exact duplicate rows
        3. **Handle missing values** - Fills or drops based on data type and missing percentage:
           - Drops columns with >50% missing values
           - Fills categorical data with mode or 'Unknown'
           - Fills numerical data with median
        4. **Optimize data types** - Converts object columns to numeric where possible
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("🚀 Run Automated Cleaning", type="primary", use_container_width=True):
                with st.spinner("Running automated cleaning..."):
                    before_shape = st.session_state.df.shape
                    st.session_state.df = automated_data_cleaning(st.session_state.df)
                    after_shape = st.session_state.df.shape
                    
                    st.success(f"""
                    ✅ Automated cleaning completed!
                    - Shape changed from {before_shape} to {after_shape}
                    - {len(st.session_state.cleaning_log)} cleaning steps performed
                    """)
                    st.rerun()
            
            st.subheader("Reset Data")
            if st.button("🔄 Reset to Original", use_container_width=True):
                st.session_state.df = st.session_state.original_df.copy()
                st.session_state.cleaning_log = []
                st.success("Dataset reset to original state!")
                st.rerun()
        
        with col2:
            if st.session_state.df is not None:
                st.subheader("Data Preview After Cleaning")
                st.dataframe(st.session_state.df.head(), use_container_width=True)
                
                # Comparison with original
                if st.session_state.original_df is not None:
                    st.subheader("Before vs After Comparison")
                    comparison_df = pd.DataFrame({
                        'Metric': ['Rows', 'Columns', 'Missing Values', 'Duplicates'],
                        'Original': [
                            st.session_state.original_df.shape[0],
                            st.session_state.original_df.shape[1],
                            st.session_state.original_df.isnull().sum().sum(),
                            st.session_state.original_df.duplicated().sum()
                        ],
                        'Current': [
                            st.session_state.df.shape[0],
                            st.session_state.df.shape[1],
                            st.session_state.df.isnull().sum().sum(),
                            st.session_state.df.duplicated().sum()
                        ]
                    })
                    comparison_df['Change'] = comparison_df['Current'] - comparison_df['Original']
                    st.dataframe(comparison_df, use_container_width=True)
    
    with tab4:
        display_cleaning_log()
        
        # Download cleaned data
        if st.session_state.df is not None:
            st.subheader("📥 Download Cleaned Data")
            
            # Convert dataframe to CSV
            csv = st.session_state.df.to_csv(index=False)
            st.download_button(
                label="Download Cleaned Dataset as CSV",
                data=csv,
                file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Generate cleaning report
            if st.session_state.cleaning_log:
                report = []
                report.append("# Data Cleaning Report")
                report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                report.append("")
                report.append("## Summary")
                report.append(f"- Original shape: {st.session_state.original_df.shape}")
                report.append(f"- Final shape: {st.session_state.df.shape}")
                report.append(f"- Number of cleaning steps: {len(st.session_state.cleaning_log)}")
                report.append("")
                report.append("## Cleaning Steps")
                report.append("")
                
                for i, log_entry in enumerate(st.session_state.cleaning_log, 1):
                    report.append(f"### {i}. {log_entry['step']}")
                    report.append(f"**Time:** {log_entry['timestamp']}")
                    report.append(f"**Shape change:** {log_entry['before_shape']} → {log_entry['after_shape']}")
                    report.append(f"**Reasoning:** {log_entry['reasoning']}")
                    report.append("**Code:**")
                    report.append(f"```python\n{log_entry['code']}\n```")
                    report.append("")
                
                report_text = "\n".join(report)
                st.download_button(
                    label="Download Cleaning Report",
                    data=report_text,
                    file_name=f"cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

else:
    st.info("👆 Please upload a CSV file in the sidebar to begin data cleaning!")
    
    # Show example of what the app can do
    st.subheader("Features of this Data Cleaning Assistant:")
    st.markdown("""
    - 📊 **Comprehensive Data Overview** - View data shape, types, missing values, and statistics
    - 🔧 **Manual Cleaning Tools** - Handle missing values, remove duplicates, drop columns with full control
    - 🤖 **Automated Cleaning** - Intelligent cleaning based on data characteristics and best practices
    - 📋 **Detailed Documentation** - Every cleaning step is logged with code and reasoning
    - 📥 **Export Options** - Download cleaned data and detailed cleaning reports
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")