import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Administrator\Desktop\New folder\accident_prediction_india.csv")  # replace with your actual dataset filename
    return df

df = load_data()

st.title("📊 Road Safety Data Visualization")

st.sidebar.header("Select Visualization")

plot_type = st.sidebar.radio(
    "Choose a plot",
    [
        "Accidents by State",
        "Accident Severity Distribution",
        "Alcohol Involvement by State",
        "Monthly Accident Trends",
        "Weather Conditions Pie"
    ]
)

if plot_type == "Accidents by State":
    st.subheader("Top States by Number of Accidents")
    state_counts = df["State Name"].value_counts().head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=state_counts.values, y=state_counts.index, ax=ax, palette="viridis")
    ax.set_xlabel("Number of Accidents")
    ax.set_ylabel("State")
    st.pyplot(fig)

elif plot_type == "Accident Severity Distribution":
    st.subheader("Distribution of Accident Severity")
    severity_counts = df["Accident Severity"].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=severity_counts.index, y=severity_counts.values, palette="coolwarm")
    ax.set_ylabel("Number of Cases")
    st.pyplot(fig)

elif plot_type == "Alcohol Involvement by State":
    st.subheader("Alcohol Involvement by State")
    alcohol_df = df[df["Alcohol Involvement"] == "Yes"]
    alcohol_state_counts = alcohol_df["State Name"].value_counts().head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=alcohol_state_counts.values, y=alcohol_state_counts.index, palette="rocket")
    ax.set_xlabel("Number of Alcohol-related Accidents")
    st.pyplot(fig)

elif plot_type == "Monthly Accident Trends":
    st.subheader("Monthly Trend of Accidents")
    if "Month" in df.columns:
        df["Month"] = df["Month"].astype(str)
        monthly_counts = df["Month"].value_counts().reindex([
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ])
        fig, ax = plt.subplots()
        sns.lineplot(x=monthly_counts.index, y=monthly_counts.values, marker="o", ax=ax)
        ax.set_ylabel("Number of Accidents")
        ax.set_xticklabels(monthly_counts.index, rotation=45)
        st.pyplot(fig)
    else:
        st.warning("Month column not found in dataset.")

elif plot_type == "Weather Conditions Pie":
    st.subheader("Weather Conditions During Accidents")
    weather_counts = df["Weather Conditions"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(weather_counts.values, labels=weather_counts.index, autopct="%1.1f%%", startangle=140)
    ax.axis("equal")
    st.pyplot(fig)
