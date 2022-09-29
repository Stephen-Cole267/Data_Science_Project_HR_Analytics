from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import streamlit as st
import os

st.set_page_config("Predict", layout="wide")


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


selected_features = [
    "EducationField",
    "JobRole",
    "MaritalStatus",
    "TotalWorkingYears",
    "EnvironmentSatisfaction",
    "YearsSinceLastPromotion",
    "BusinessTravel",
    "Department",
]


@st.cache
def create_encoders(dataframe: pd.DataFrame):
    transformer = make_column_transformer(
        (
            OneHotEncoder(),
            [
                "EducationField",
                "JobRole",
                "MaritalStatus",
                "BusinessTravel",
                "Department",
            ],
        ),
        remainder="passthrough",
    )
    transformed = transformer.fit_transform(dataframe)
    # transformed_df = pd.DataFrame(transformed)
    transformed_df: pd.DataFrame = pd.DataFrame.sparse.from_spmatrix(transformed)
    transformed_df.columns = transformer.get_feature_names_out()
    return transformer, transformed_df


def train_predicting_model():
    input_df = data_df.copy()
    # only select features
    final_X = input_df[selected_features]
    final_y = input_df["Attrition"]
    transformer, final_X = create_encoders(final_X)

    # train
    final_model = XGBClassifier()
    final_model.fit(final_X, final_y)
    return transformer, final_model

    # return final_df


final_transformer, final_model = train_predicting_model()


def get_prediction(
    education_field: str,
    job_role: str,
    marital_status: str,
    total_working_years: str,
    environment_satisfaction: str,
    years_since_last_promotion: str,
    business_travel: str,
    department: str,
):
    # create a dataframe
    input_df = pd.DataFrame(
        {
            "EducationField": [education_field],
            "JobRole": [job_role],
            "MaritalStatus": [marital_status],
            "TotalWorkingYears": [total_working_years],
            "EnvironmentSatisfaction": [environment_satisfaction],
            "YearsSinceLastPromotion": [years_since_last_promotion],
            "BusinessTravel": [business_travel],
            "Department": [department],
        }
    )
    # transform
    transformed = final_transformer.transform(input_df)
    # transformed_df = pd.DataFrame(transformed)
    transformed_df: pd.DataFrame = pd.DataFrame.sparse.from_spmatrix(transformed)
    transformed_df.columns = final_transformer.get_feature_names_out()
    # predict
    y_pred = final_model.predict(transformed_df)
    return y_pred[0]


st.title("Predict Employee Attrition")


with st.form("predict_form"):
    col1, col2, col3, col4 = st.columns(4)

    education_field = col1.selectbox(
        "Education Field", data_df["EducationField"].sort_values().unique(), index=0
    )
    job_role = col2.selectbox(
        "Job Role", data_df["JobRole"].sort_values().unique(), index=0
    )
    marital_status = col3.selectbox(
        "Marital Status", data_df["MaritalStatus"].sort_values().unique(), index=0
    )
    total_working_years = col4.number_input(
        "Total Working Years",
        min_value=0,
        max_value=int(data_df["TotalWorkingYears"].max()),
        step=1,
        value=0,
    )
    environment_satisfaction = col1.selectbox(
        "Environment Satisfaction",
        data_df["EnvironmentSatisfaction"].sort_values().unique(),
        index=0,
    )

    years_since_last_promotion = col2.number_input(
        "Years Since Last Promotion",
        min_value=0,
        max_value=int(data_df["YearsSinceLastPromotion"].max()),
        step=1,
        value=0,
    )
    business_travel = col3.selectbox(
        "Business Travel", data_df["BusinessTravel"].sort_values().unique(), index=0
    )

    department = col4.selectbox(
        "Department", data_df["Department"].sort_values().unique(), index=0
    )

    predict_button = st.form_submit_button("Predict")

if predict_button:
    prediction = get_prediction(
        education_field,
        job_role,
        marital_status,
        total_working_years,
        environment_satisfaction,
        years_since_last_promotion,
        business_travel,
        department,
    )
    prediction_text = "Will leave" if prediction == 1 else "Will not leave"
    st.write(f"## Prediction: {prediction_text}")
