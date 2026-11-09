
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI-Powered House Price Prediction System | Analytics Dashboard",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 2. LOAD MODEL
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "house_price_app_artifacts.joblib"


@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH)


try:
    artifacts = load_artifacts()
    model = artifacts["model"]
    features = artifacts["features"]

except FileNotFoundError:
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()


# ==================================================
# 3. COLOURFUL DESIGN
# ==================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f3f4ff 0%, #f8fbff 55%, #eefbfa 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #17153b 0%, #312e81 55%, #4338ca 100%);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="stSidebar"] .stRadio label {
    font-weight: 600;
}

h1 {
    color: #17153b;
    font-weight: 800;
}

h2, h3 {
    color: #312e81;
    font-weight: 700;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffff, #eef2ff);
    border: 1px solid #dbe3ff;
    border-radius: 15px;
    padding: 18px;
    box-shadow: 0 5px 15px rgba(49, 46, 129, 0.08);
}

[data-testid="stMetricLabel"] {
    color: #4b5563;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #312e81;
    font-weight: 800;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    height: 48px;
    font-weight: 700;
    transition: 0.3s;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(90deg, #3730a3, #6d28d9);
    color: white;
    border: none;
}

div[data-testid="stForm"] {
    background: rgba(255,255,255,0.85);
    border: 1px solid #dbe3ff;
    border-radius: 16px;
    padding: 22px;
}

hr {
    border-color: #c7d2fe;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 4. HELPER FUNCTIONS
# ==================================================

def section_header(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def format_money(value):
    return f"${value:,.2f}"


# ==================================================
# 5. SIDEBAR
# ==================================================

st.sidebar.markdown("# 🏡 House Price AI")
st.sidebar.markdown("### Machine Learning Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "Home",
        "Price Prediction",
        "Model Performance",
        "About"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**MODEL INFORMATION**")
st.sidebar.write("Algorithm: Random Forest")
st.sidebar.write(f"Features: {len(features)}")
st.sidebar.write("Dataset: Ames Housing")

st.sidebar.markdown("---")
st.sidebar.caption("Developed by Gloria Nkum")


# ==================================================
# 6. HOME PAGE
# ==================================================

if page == "Home":

    st.markdown("# 🏡 House Price Prediction")
    st.markdown(
        "### Intelligent Property Valuation & Machine Learning Dashboard"
    )

    st.markdown(
        "Explore property values, generate price estimates and "
        "understand the performance of a machine learning model."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "🤖 Machine Learning Model",
        "Random Forest"
    )

    col2.metric(
        "📊 Model Features",
        len(features)
    )

    col3.metric(
        "🎯 R² Score",
        f"{artifacts['r2']:.4f}"
    )

    st.markdown("---")

    section_header(
        "Project Overview",
        "A machine learning solution for residential property price estimation."
    )

    left, right = st.columns([1.4, 1])

    with left:
        st.markdown("""
        This application uses a **Random Forest Regression model**
        trained on the Ames Housing dataset.

        The model analyses 12 numerical property characteristics
        to estimate residential house prices.

        **What can you do here?**

        - Generate house price estimates.
        - Explore model performance.
        - Examine feature importance.
        - Understand prediction errors.
        """)

    with right:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg,#4338ca,#7c3aed,#0d9488);
            padding: 30px;
            border-radius: 18px;
            color: white;
            text-align: center;
        ">
            <h2 style="color:white;">AI POWERED</h2>
            <h1 style="color:white;font-size:48px;">🏠</h1>
            <h3 style="color:white;">Property Intelligence</h3>
            <p>Data-driven house price estimation</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.info(
        "Dataset prices are expressed in US dollars (USD). "
        "This model is based on historical Ames, Iowa housing data, "
        "not Ghanaian property market data."
    )


# ==================================================
# 7. PRICE PREDICTION PAGE
# ==================================================

elif page == "Price Prediction":

    st.title("🏡 Property Price Estimator")

    st.write(
        "Enter the property characteristics to estimate its market price."
    )

    st.markdown("---")

    with st.form("prediction_form"):

        st.markdown("### 🏠 Property Quality and Size")

        col1, col2 = st.columns(2)

        with col1:

            overall_qual = st.slider(
                "Overall Quality",
                1, 10, 5,
                help="Overall material and finish quality."
            )

            gr_liv_area = st.number_input(
                "Above Ground Living Area (sq ft)",
                334, 5642, 1500
            )

            first_floor = st.number_input(
                "First Floor Area (sq ft)",
                334, 4692, 1000
            )

            second_floor = st.number_input(
                "Second Floor Area (sq ft)",
                0, 2065, 500
            )

            total_basement = st.number_input(
                "Total Basement Area (sq ft)",
                0, 6110, 800
            )

            lot_area = st.number_input(
                "Lot Area (sq ft)",
                1300, 215245, 8000
            )

        with col2:

            total_rooms = st.number_input(
                "Total Rooms Above Ground",
                2, 14, 6
            )

            full_bath = st.number_input(
                "Full Bathrooms",
                0, 3, 2
            )

            garage_cars = st.number_input(
                "Garage Capacity",
                0, 4, 2
            )

            garage_area = st.number_input(
                "Garage Area (sq ft)",
                0, 1418, 500
            )

            year_built = st.number_input(
                "Year Built",
                1872, 2010, 2000
            )

            year_remod = st.number_input(
                "Year Remodelled",
                1872, 2010, 2005
            )

        st.markdown("---")

        submitted = st.form_submit_button(
            "✨ GENERATE HOUSE PRICE PREDICTION",
            use_container_width=True
        )

    if submitted:

        if year_remod < year_built:

            st.error(
                "The remodelling year cannot be earlier than the "
                "construction year."
            )

        else:

            input_data = pd.DataFrame([{
                "OverallQual": overall_qual,
                "YearBuilt": year_built,
                "GrLivArea": gr_liv_area,
                "1stFlrSF": first_floor,
                "TotalBsmtSF": total_basement,
                "2ndFlrSF": second_floor,
                "FullBath": full_bath,
                "TotRmsAbvGrd": total_rooms,
                "GarageCars": garage_cars,
                "GarageArea": garage_area,
                "LotArea": lot_area,
                "YearRemodAdd": year_remod
            }])

            input_data = input_data[features]

            prediction = model.predict(input_data)[0]

            st.markdown("---")

            st.success("Your prediction has been generated!")

            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(120deg,#312e81,#6d28d9,#0891b2);
                    padding: 30px;
                    border-radius: 18px;
                    text-align: center;
                    color: white;
                ">
                    <h3 style="color:white;">ESTIMATED HOUSE PRICE</h3>
                    <h1 style="color:white;font-size:42px;">
                        ${prediction:,.2f}
                    </h1>
                    <p style="color:white;">
                        Random Forest Regression Prediction
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("---")

            section_header("Property Summary")

            st.dataframe(
                input_data.T.rename(columns={0: "Input Value"}),
                use_container_width=True
            )

            st.caption(
                "This is a model-based estimate, not a formal property valuation."
            )


# ==================================================
# 8. MODEL PERFORMANCE PAGE
# ==================================================

elif page == "Model Performance":

    st.title("📊 Model Performance Analytics")

    st.write(
        "Explore the evaluation results and predictive behaviour "
        "of the Random Forest model."
    )

    st.markdown("---")

    # KPI METRICS

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "💰 Mean Absolute Error",
            format_money(artifacts["mae"])
        )

    with col2:
        st.metric(
            "📉 Root Mean Squared Error",
            format_money(artifacts["rmse"])
        )

    with col3:
        st.metric(
            "🎯 R² Score",
            f"{artifacts['r2']:.4f}",
            help="Proportion of variation in test prices explained by the model."
        )

    st.markdown("---")

    # PREPARE DATA

    actual = pd.Series(artifacts["actual_prices"])
    predicted = pd.Series(artifacts["predicted_prices"])

    chart_data = pd.DataFrame({
        "Actual Price": actual,
        "Predicted Price": predicted
    })

    chart_data["Prediction Error"] = (
        chart_data["Actual Price"] - chart_data["Predicted Price"]
    )

    chart_data["Absolute Error"] = (
        chart_data["Prediction Error"].abs()
    )

    # ACTUAL VS PREDICTED

    section_header(
        "Actual vs Predicted House Prices",
        "Each point represents one house in the held-out test dataset."
    )

    fig = px.scatter(
        chart_data,
        x="Actual Price",
        y="Predicted Price",
        color="Absolute Error",
        color_continuous_scale="Turbo",
        hover_data={
            "Actual Price": ":,.0f",
            "Predicted Price": ":,.0f",
            "Prediction Error": ":,.0f",
            "Absolute Error": ":,.0f"
        },
        title="Actual Prices Compared with Model Predictions"
    )

    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())

    fig.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[lower, upper],
            mode="lines",
            name="Perfect Prediction",
            line=dict(
                color="#111827",
                dash="dash",
                width=2
            )
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=550,
        font=dict(color="#1e1b4b"),
        title_font=dict(size=20),
        xaxis_title="Actual Price (USD)",
        yaxis_title="Predicted Price (USD)",
        coloraxis_colorbar=dict(title="Absolute Error")
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Points closer to the diagonal line represent more accurate predictions."
    )

    st.markdown("---")

    # FEATURE IMPORTANCE

    section_header(
        "Feature Importance",
        "Which property characteristics contributed most to the model's predictions?"
    )

    importance = artifacts["feature_importance"].copy()

    importance = importance.sort_values(
        "Importance",
        ascending=True
    )

    feature_fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Plasma",
        title="Relative Importance of Property Features",
        text="Importance"
    )

    feature_fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    feature_fig.update_layout(
        template="plotly_white",
        height=600,
        font=dict(color="#1e1b4b"),
        showlegend=False,
        coloraxis_showscale=False,
        xaxis_title="Relative Feature Importance",
        yaxis_title="Property Feature"
    )

    st.plotly_chart(feature_fig, use_container_width=True)

    st.caption(
        "Feature importance indicates relative contribution within the "
        "trained Random Forest model. It does not establish causation."
    )

    st.markdown("---")

    # PREDICTION ERROR DISTRIBUTION

    section_header(
        "Prediction Error Distribution",
        "Understand how far the model's predictions differ from actual prices."
    )

    error_fig = px.histogram(
        chart_data,
        x="Prediction Error",
        nbins=35,
        color_discrete_sequence=["#8b5cf6"],
        title="Distribution of Prediction Errors"
    )

    error_fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="Zero Error",
        annotation_position="top"
    )

    error_fig.update_layout(
        template="plotly_white",
        height=450,
        font=dict(color="#1e1b4b"),
        xaxis_title="Actual Price - Predicted Price (USD)",
        yaxis_title="Number of Houses"
    )

    st.plotly_chart(error_fig, use_container_width=True)

    st.caption(
        "Positive errors indicate underprediction; negative errors indicate "
        "overprediction."
    )

    st.markdown("---")

    # ERROR SUMMARY

    section_header("Prediction Error Summary")

    error_col1, error_col2, error_col3 = st.columns(3)

    error_col1.metric(
        "Average Signed Error",
        format_money(chart_data["Prediction Error"].mean())
    )

    error_col2.metric(
        "Largest Absolute Error",
        format_money(chart_data["Absolute Error"].max())
    )

    error_col3.metric(
        "Median Absolute Error",
        format_money(chart_data["Absolute Error"].median())
    )


# ==================================================
# 9. ABOUT PAGE
# ==================================================

elif page == "About":

    st.title("ℹ️ About This Project")

    st.markdown("""
    ## House Price Prediction Using Machine Learning

    This project demonstrates an end-to-end machine learning
    regression workflow.

    ### Project Objective

    To develop a model that estimates residential house prices
    using property characteristics.

    ### Dataset

    Ames Housing Dataset.

    ### Machine Learning Algorithm

    Random Forest Regression.

    ### Model Inputs

    The model uses 12 numerical features, including:

    - Overall Quality
    - Year Built
    - Living Area
    - Basement Area
    - Garage Capacity
    - Lot Area
    - Number of Rooms

    ### Technologies

    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Matplotlib
    - Plotly
    - Streamlit
    - Joblib

    ### Evaluation Metrics

    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - R-squared (R²)

    ### Developer

    Gloria Nkum

    **Data Science | Machine Learning | Analytics**
    """)

    st.info(
        "This application is an educational and portfolio project. "
        "Its predictions should not be treated as professional property valuations."
    )


# ==================================================
# 10. FOOTER
# ==================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#64748b;padding:10px;">
        <b>AI-Powered House Price Prediction System</b> |
        Machine Learning Portfolio Project |
        Built with Python & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
