import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import os

pio.templates.default = "seaborn"
st.set_page_config(layout="wide")


@st.cache
def read_data() -> pd.DataFrame:
    conn = (
        os.getenv("DATABASE_URL") or "postgresql://postgres:pass@localhost/hr_database"
    )
    employee_df = pd.read_sql_table("employee_survey_data", conn)
    general_df = pd.read_sql_table("general_data", conn)
    manager_df = pd.read_sql_table("manager_survey_data", conn)

    data_df = general_df.merge(employee_df, on="EmployeeID", how="left")
    data_df = data_df.merge(manager_df, on="EmployeeID", how="left")

    in_time_df = pd.read_csv("Datasets/in_time.csv")
    out_time_df = pd.read_csv("Datasets/out_time.csv")

    # rename first column
    in_time_df.rename(columns={"Unnamed: 0": "EmployeeID"}, inplace=True)
    # change index to EmployeeID
    in_time_df.set_index("EmployeeID", inplace=True)
    # change dtype to datetime
    in_time_df = in_time_df.apply(pd.to_datetime)

    # repeat for out_time_df
    out_time_df.rename(columns={"Unnamed: 0": "EmployeeID"}, inplace=True)
    out_time_df.set_index("EmployeeID", inplace=True)
    out_time_df = out_time_df.apply(pd.to_datetime)

    # create df with in_time and out_time difference
    time_diff_df = out_time_df - in_time_df

    # calculate average time ignoring NaN values
    time_diff_df["AverageTime"] = time_diff_df.mean(axis=1, skipna=True)
    # calculate total time ignoring NaN values
    time_diff_df["TotalTime"] = time_diff_df.sum(axis=1, skipna=True)
    # drop all columns except Average and Total
    time_diff_df = time_diff_df[["AverageTime", "TotalTime"]]
    # convert to seconds
    time_diff_df = time_diff_df.apply(lambda x: x / np.timedelta64(1, "s"))

    time_diff_df = time_diff_df.reset_index()

    # join time_diff_df with data_df
    data_df = data_df.merge(time_diff_df, on="EmployeeID", how="left")

    data_df.columns = data_df.columns.astype(str)

    # drop rows with missing values
    data_df_no_na = data_df.dropna()

    # ordered categorical data to numerical
    data_df_no_na["Attrition"] = data_df_no_na["Attrition"].map({"Yes": 1, "No": 0})
    # data_df["Over18"] = data_df["Over18"].map({"Y": 1, "N": 0})
    data_df_no_na["BusinessTravel"] = data_df_no_na["BusinessTravel"].map(
        {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
    )

    return data_df_no_na


data_df = read_data()

ignored_columns = [
    "EmployeeID",
    "Over18",
    "StandardHours",
]

numerical_columns = [
    # "EmployeeID",
    "Age",
    # "Attrition",
    # "BusinessTravel",
    # "Department",
    "DistanceFromHome",
    # "Education",
    # "EducationField",
    "EmployeeCount",
    # "Gender",
    # "JobLevel",
    # "JobRole",
    # "MaritalStatus",
    "MonthlyIncome",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "YearsAtCompany",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    # 'EnvironmentSatisfaction',
    # 'JobSatisfaction',
    # 'WorkLifeBalance',
    # 'JobInvolvement',
    # 'PerformanceRating'
    "AverageTime",
    "TotalTime",
]

categorical_columns = [
    var
    for var in data_df.columns
    if var not in numerical_columns and var not in ignored_columns
]


def compare_categorical_vis(data, variable):
    fig = px.histogram(
        data,
        x=variable,
        color="Attrition",
        barmode="overlay",
        title=f"Histogram of {variable} by Attrition",
    )
    # bargap
    fig.update_layout(bargap=0.1)
    return fig


def compare_numerical_vis(data, variable):
    hist_fig = px.histogram(data, x=variable, color="Attrition")

    box_fig = px.box(data, x="Attrition", y=variable, color="Attrition")

    final_fig = make_subplots(rows=1, cols=2)
    final_fig.add_trace(hist_fig.data[0], row=1, col=1)
    final_fig.add_trace(hist_fig.data[1], row=1, col=1)

    final_fig.add_trace(box_fig.data[0], row=1, col=2)
    final_fig.add_trace(box_fig.data[1], row=1, col=2)
    # hide legend
    # final_fig.update_layout(showlegend=False)
    # title
    final_fig.update_layout(title_text=f"Attrition by {variable}")
    # title center
    final_fig.update_layout(title_x=0.5)
    # barmode
    final_fig.update_layout(barmode="overlay")
    return final_fig


st.title("Variable visualisation")

selected_variable = st.sidebar.selectbox(
    "Select a variable", numerical_columns + categorical_columns
)


st.write(data_df.head())

# st.write("## Data Visualisation")
if selected_variable in categorical_columns:
    st.plotly_chart(
        compare_categorical_vis(data_df, selected_variable), use_container_width=True
    )
else:
    st.plotly_chart(
        compare_numerical_vis(data_df, selected_variable), use_container_width=True
    )
