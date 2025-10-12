import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Set the page configuration for the Streamlit app
st.set_page_config(page_title="House Price Prediction", layout="wide")

# --- Custom CSS for Theme ---
st.markdown(
    """
<style>
    /* Sidebar background and text color */
    [data-testid="stSidebar"] {
        background-color: #FFD700; /* Yellow */
        color: #000000; /* Black text */
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- Data Loading and Caching ---
@st.cache_data
def load_data():
    """Loads the housing price dataset from a CSV file."""
    try:
        df = pd.read_csv("housing_price_dataset.csv")
        return df
    except FileNotFoundError:
        st.error(
            "The 'housing_price_dataset.csv' file was not found. "
            "Please make sure it's in the same directory as the app."
        )
        return None


df = load_data()

# --- Model Training ---
@st.cache_resource
def train_svm_model(df):
    """Trains a Support Vector Machine (SVM) regressor model."""
    if df is None or df.empty:
        return None, None, None

    # Feature Engineering
    current_year = datetime.now().year
    if "YearBuilt" not in df.columns:
        st.warning("Column 'YearBuilt' missing. Cannot compute 'Age'.")
        return None, None, None

    df = df.copy()
    df["Age"] = current_year - df["YearBuilt"]

    # One-hot encode Neighborhood (drop_first to avoid perfect multicollinearity)
    if "Neighborhood" in df.columns:
        df = pd.get_dummies(df, columns=["Neighborhood"], drop_first=True)

    # Define features (X) and target (y)
    features = ["SquareFeet", "Bedrooms", "Bathrooms", "Age"] + [
        col for col in df.columns if col.startswith("Neighborhood_")
    ]

    # Validate required columns exist
    missing = [c for c in features + ["Price"] if c not in df.columns]
    if missing:
        st.warning(f"Missing required columns for model training: {missing}")
        return None, None, None

    X = df[features]
    y = df["Price"]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train_scaled, _, y_train, _ = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train the SVM model
    svm = SVR(C=100, gamma=0.1, kernel="rbf")
    svm.fit(X_train_scaled, y_train)

    return svm, scaler, features


svm_model, scaler, model_features = train_svm_model(df.copy() if df is not None else None)

# --- Streamlit App Layout ---

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Exploratory Data Analysis", "Prediction"])

# --- Home Page ---
if page == "Home":
    st.title("🏡 Housing Price Dataset")
    if df is None:
        st.info("No dataset loaded. Place 'housing_price_dataset.csv' next to this app.")
    else:
        st.write("This dataset contains information about houses, including size, rooms, location, and price.")
        st.dataframe(df)

# --- EDA Page ---
elif page == "Exploratory Data Analysis":
    st.title("🔍 Exploratory Data Analysis")
    if df is None:
        st.info("No dataset loaded. Place 'housing_price_dataset.csv' next to this app.")
    else:
        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        df_encoded = pd.get_dummies(df, columns=["Neighborhood"]) if "Neighborhood" in df.columns else df.copy()
        corr_matrix = df_encoded.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap="RdYlBu", fmt=".2f", ax=ax)
        st.pyplot(fig)

        # Box Plots
        st.subheader("Box Plots of Numerical Features")
        numerical_columns = ["SquareFeet", "Bedrooms", "Bathrooms", "YearBuilt", "Price"]
        available_cols = [c for c in numerical_columns if c in df.columns]
        if not available_cols:
            st.info("No required numerical columns found for box plots.")
        else:
            plot_df = df[available_cols].dropna()
            if plot_df.empty:
                st.info("No non-NaN data available for the selected numerical columns.")
            else:
                fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
                axes = axes.flatten()
                for i, col in enumerate(available_cols):
                    sns.boxplot(y=plot_df[col], ax=axes[i], color="#D3C30F")
                    axes[i].set_title(f"Box Plot of {col}")
                    axes[i].set_ylabel(col)
                for j in range(len(available_cols), len(axes)):
                    fig.delaxes(axes[j])
                plt.tight_layout()
                st.pyplot(fig)

        # Distribution Plots
        st.subheader("Distribution of Numerical Features")
        available_cols = [c for c in numerical_columns if c in df.columns]
        if available_cols:
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 10))
            fig.suptitle("Distribution of Numerical Features in the Housing Dataset", fontsize=16)
            axes = axes.flatten()
            plot_df = df[available_cols].dropna()
            for i, col in enumerate(available_cols):
                sns.histplot(plot_df[col], kde=True, ax=axes[i], bins=30, color="red")
                axes[i].set_title(f"Distribution of {col}")
                axes[i].set_xlabel(col)
                axes[i].set_ylabel("Frequency")
            for j in range(len(available_cols), len(axes)):
                fig.delaxes(axes[j])
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            st.pyplot(fig)
        else:
            st.info("No required numerical columns found for distribution plots.")

        # Pair Plot
        st.subheader("Pair Plot of Key Features")
        st.write("This plot shows pairwise relationships between features, colored by Neighborhood.")
        pair_cols = ["SquareFeet", "Bedrooms", "Bathrooms", "YearBuilt", "Price"]
        pair_cols = [c for c in pair_cols if c in df.columns]
        try:
            if "Neighborhood" in df.columns and len(pair_cols) >= 2:
                pairgrid = sns.pairplot(df[pair_cols + ["Neighborhood"]], hue="Neighborhood", palette="YlGnBu")
                st.pyplot(pairgrid.fig)
            elif len(pair_cols) >= 2:
                pairgrid = sns.pairplot(df[pair_cols])
                st.pyplot(pairgrid.fig)
            else:
                st.info("Not enough columns available to create a pair plot.")
        except Exception as e:
            st.warning(f"Could not generate pair plot: {e}")

        # Scatter grid of features vs Price
        st.subheader("Feature Relationships with House Price")
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 12))
        fig.suptitle("Feature Relationships with House Price", fontsize=16)
        axes = axes.flatten()
        # Prepare columns if available
        def safe_scatter(col_idx, x_col, title, color):
            if x_col in df.columns and "Price" in df.columns:
                sns.scatterplot(data=df, x=x_col, y="Price", ax=axes[col_idx], alpha=0.5, color=color)
                axes[col_idx].set_title(title)
            else:
                axes[col_idx].set_visible(False)

        safe_scatter(0, "SquareFeet", "SquareFeet vs. Price", "green")
        safe_scatter(1, "Bedrooms", "Bedrooms vs. Price", "#FF6A00")
        safe_scatter(2, "Bathrooms", "Bathrooms vs. Price", "#0010F6")
        safe_scatter(3, "YearBuilt", "YearBuilt vs. Price", "#E20B0B")

        # Neighborhood vs Price box (if available)
        if "Neighborhood" in df.columns and "Price" in df.columns:
            sns.boxplot(data=df, x="Neighborhood", y="Price", ax=axes[4], palette=["#05B4FF", "#BCFF05", "#DB8428"])
            axes[4].set_title("Neighborhood vs. Price")
        else:
            axes[4].set_visible(False)

        # Hide the last subplot if unused
        if len(axes) > 5:
            fig.delaxes(axes[5])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(fig)


# --- Prediction Page ---
elif page == "Prediction":
    st.title("💰 House Price Prediction")
    if svm_model is None or scaler is None or model_features is None:
        st.warning("Model is not available. Ensure dataset has required columns and the model trained successfully.")
    else:
        st.write("Use the sliders and dropdown to set the features of the house and get a price prediction.")

        # User input for house features
        square_feet = st.slider("Square Feet", 500, 5000, 2000)
        bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5])
        bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4])
        neighborhood = st.selectbox("Neighborhood", ["Rural", "Suburb", "Urban"])
        year_built = st.slider("Year Built", 1900, datetime.now().year, 1985)

        # Prediction button
        if st.button("Predict Price"):
            # Create a DataFrame for the user's input
            input_data = pd.DataFrame(
                {
                    "SquareFeet": [square_feet],
                    "Bedrooms": [bedrooms],
                    "Bathrooms": [bathrooms],
                    "Age": [datetime.now().year - year_built],
                    # Create the common neighborhood one-hot columns used by the model
                    "Neighborhood_Suburb": [1 if neighborhood == "Suburb" else 0],
                    "Neighborhood_Urban": [1 if neighborhood == "Urban" else 0],
                }
            )

            # Ensure all feature columns are present, adding missing ones with a value of 0
            for col in model_features:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[model_features]

            # Scale the input data
            try:
                input_data_scaled = scaler.transform(input_data)
                # Make the prediction
                prediction = svm_model.predict(input_data_scaled)
                st.success(f"The predicted price of the house is: ${prediction[0]:,.2f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# Fallback: if none of the conditions match
else:
    st.info("Select a page from the sidebar to get started.")