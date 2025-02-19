import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import models
import models_balanced

@st.cache_data
def load_data():
    df = pd.read_csv("predictive_maintenance.csv")
    return df

st.set_page_config(page_title="Predictive Fault Detection", layout="wide")
st.title("Predictive Fault Detection (AMS100)")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select page", 
    [
        "Q2: Data Inspection & Profiling", 
        "Q3: Interactive Data Visualization",
        "Q4: Model Performance",
        "Q6: Model Performance on Augmented Dataset"
    ]
)

st.sidebar.markdown("---")
st.sidebar.image("footer.png", use_column_width=True)

df = load_data()

# ---------------------------
# Q2: Data Inspection & Profiling
if page == "Q2: Data Inspection & Profiling":
    st.header("Data Inspection & Profiling")
    st.markdown("Below is a preview of the dataset for the Predictive Fault Detection application.")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Filter Data by Failure Type")
    failure_types = df["Failure Type"].unique().tolist()
    selected_failures = st.multiselect("Select Failure Type(s):", failure_types, default=failure_types)
    filtered_df = df[df["Failure Type"].isin(selected_failures)]
    st.write(f"Number of observations after filtering: {filtered_df.shape[0]}")
    st.dataframe(filtered_df.head(10), use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.write(filtered_df.describe())

    st.subheader("Correlation Matrix")
    numeric_columns = filtered_df.select_dtypes(include=np.number).columns
    corr_matrix = filtered_df[numeric_columns].corr()

    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

    st.subheader("Attribute Distributions")
    selected_feature = st.selectbox("Select a feature to visualize", filtered_df.columns)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[selected_feature], kde=True, ax=ax)
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=df[selected_feature], ax=ax)
    st.pyplot(fig)
    
# ---------------------------
# Q3: Interactive Data Visualization
elif page == "Q3: Interactive Data Visualization":
    st.header("Interactive Data Visualization")
    st.markdown("This section provides different visualization tools to explore the dataset.")
    plot_type = st.selectbox("Select Plot Type:", ["Box Plot", "Pair Plot"])#["Scatter Plot", "Histogram", "Box Plot", "Pair Plot"])

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    # if plot_type == "Scatter Plot":
    #     st.subheader("Scatter Plot")
    #     x_axis = st.selectbox("Choose X-axis:", numeric_columns, index=0)
    #     y_axis = st.selectbox("Choose Y-axis:", numeric_columns, index=1)
        
    #     fig = px.scatter(df, x=x_axis, y=y_axis,
    #                     title=f"Scatter Plot of {y_axis} vs. {x_axis}")
    #     st.plotly_chart(fig, use_container_width=True)

    # elif plot_type == "Histogram":
    #     st.subheader("Histogram")
    #     x_axis = st.selectbox("Choose attribute for histogram:", numeric_columns, index=0)
        
    #     if x_axis:
    #         min_val = df[x_axis].min()
    #         max_val = df[x_axis].max()
    #         range_val = max_val - min_val
    #         num_bins = int(np.ceil(range_val / (range_val / 30)))  # Dynamic bin calculation
            
    #         fig = px.histogram(df, x=x_axis, nbins=num_bins,
    #                         title=f"Histogram of {x_axis} ({num_bins} bins)")
    #         st.plotly_chart(fig, use_container_width=True)

    if plot_type == "Box Plot":
        st.subheader("Box Plot")
        y_axis = st.selectbox("Choose Numeric Attribute:", numeric_columns, index=0)
        x_axis = "Failure Type"
        
        fig = px.box(df, x=x_axis, y=y_axis, title=f"Box Plot of {y_axis} by {x_axis}")
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Pair Plot":
        st.subheader("Pair Plot")
        sample_size = st.slider("Sample Size (for performance):", min_value=500, max_value=len(df), value=1000)
        sampled_df = df.sample(sample_size, random_state=42)
        
        fig = sns.pairplot(sampled_df, hue="Failure Type", diag_kind="kde", corner=True)
        st.pyplot(fig)

    
    st.subheader("Possibility of Failure")
    st.markdown("This visualization shows the possibility of failure with respect to different features.")

    def feat_prob(feature, data):
        x_vals, y_vals = [], []
        for val in sorted(data[feature].unique()):
            temp = data[data[feature] >= val]
            y_vals.append(round((temp['Target'].mean() * 100), 2))
            x_vals.append(val)
        return x_vals, y_vals

    features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    fig, axes = plt.subplots(3, 2, figsize=(15, 17))
    axes = axes.ravel()

    for i, feature in enumerate(features):
        x, y = feat_prob(feature, df)
        sns.lineplot(x=x, y=y, ax=axes[i])
        axes[i].set_title(f"Possibility of failure wrt {feature}")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Possibility of Failure (%)")
    if len(features) % 2 != 0:
        fig.delaxes(axes[-1])

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Data Filtering")
    st.markdown("Filter observations by attribute")
    feature_filter = st.selectbox("Choose Attribute:", features, index=0)
    tool_wear_min = float(df[feature_filter].min())
    tool_wear_max = float(df[feature_filter].max())
    tool_wear_range = st.slider("Select range:", tool_wear_min, tool_wear_max, (tool_wear_min, tool_wear_max))
    filtered_data = df[(df[feature_filter] >= tool_wear_range[0]) & (df[feature_filter] <= tool_wear_range[1])]
    st.write(f"Observations in selected range: {filtered_data.shape[0]}")
    st.dataframe(filtered_data.head(10), use_container_width=True)

# ---------------------------
elif page == "Q4: Model Performance":
    model_list = ["Random Forest","Logistic Regression", "Decision Tree", "Multi-Layer Perceptron", "K-Nearest Neighbors", "Support Vector Machine"]

    st.header("Model Performance")
    st.markdown("This section shows the performance of various machine learning models by displaying their classification reports and corresponding heatmaps.")
    index_links = " | ".join([f"[{model}](#{model.replace(' ', '-').lower()})" for model in model_list])
    st.markdown(index_links, unsafe_allow_html=True)

    if "classification_reports" not in st.session_state:
        with st.spinner("Training models and generating reports..."):
            st.session_state.classification_reports = models.run_models()
        st.success("Models trained and reports generated!")

    for model_name in model_list:
        st.markdown(f"<a id='{model_name.replace(' ', '-').lower()}'></a>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='font-size: 28px;'>{model_name} Performance</h2>", unsafe_allow_html=True)

        report = st.session_state.classification_reports.get(model_name, {})
        if report:
            df_report = pd.DataFrame(report).transpose()
            fig = models.get_classification_report_heatmap(report, model_name=model_name)
            st.pyplot(fig)
        else:
            st.warning("No report found for this model.")

# ---------------------------
elif page == "Q6: Model Performance on Augmented Dataset":
    model_list = ["Random Forest","Logistic Regression", "Decision Tree", "Multi-Layer Perceptron", "K-Nearest Neighbors", "Support Vector Machine"]

    st.header("Model Performance on Augmented Dataset")
    st.markdown("This section shows the performance of various machine learning models by displaying their classification reports and corresponding heatmaps.")

    index_links = " | ".join([f"[{model}](#{model.replace(' ', '-').lower()})" for model in model_list])
    st.markdown(index_links, unsafe_allow_html=True)

    if "classification_reports" not in st.session_state:
        with st.spinner("Training models and generating reports..."):
            st.session_state.classification_reports = models_balanced.run_models()
        st.success("Models trained and reports generated!")

    for model_name in model_list:
        st.markdown(f"<a id='{model_name.replace(' ', '-').lower()}'></a>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='font-size: 28px;'>{model_name} Performance</h2>", unsafe_allow_html=True)

        report = st.session_state.classification_reports.get(model_name, {})
        if report:
            df_report = pd.DataFrame(report).transpose()
            fig = models_balanced.get_classification_report_heatmap(report, model_name=model_name)
            st.pyplot(fig)
        else:
            st.warning("No report found for this model.")
