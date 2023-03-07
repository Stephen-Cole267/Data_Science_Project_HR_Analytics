"""
This script loads and cleans the data from the csv files and adds features to the data.
"""

import numpy as np
import pandas as pd


def load_general_data(verbose: bool = False) -> pd.DataFrame:
    """Load general data from csv file and return dataframe."""

    # Read data from csv file
    df = pd.read_csv("../../Datasets/general_data.csv")

    if verbose:
        # Look at the info
        print(df.info())
        # Are there any nulls?
        print(df.isnull().sum())
        # NumCompaniesWorked has 19 nulls; TotalWorkingYears has 9 nulls

    # Drop the nulls
    df.dropna(inplace=True)

    return df


def load_employee_survey_data(verbose: bool = False) -> pd.DataFrame:
    """Load employee survey data from csv file and return dataframe."""

    # Read data from csv file
    df = pd.read_csv("../../Datasets/employee_survey_data.csv")

    if verbose:
        # Look at the info
        print(df.info())
        # Are there any nulls?
        print(df.isnull().sum())

    # Drop nulls
    df.dropna(inplace=True)

    # Change cols to ints
    df["EnvironmentSatisfaction"] = df["EnvironmentSatisfaction"].astype(int)
    df["JobSatisfaction"] = df["JobSatisfaction"].astype(int)
    df["WorkLifeBalance"] = df["WorkLifeBalance"].astype(int)

    if verbose:
        # Check if EmployeeID is unique
        print(df["EmployeeID"].is_unique)  # True
    return df


def load_manager_survey_data(verbose: bool = False) -> pd.DataFrame:
    """Load manager survey data from csv file and return dataframe."""

    # Read data from csv file
    df = pd.read_csv("../../Datasets/manager_survey_data.csv")

    if verbose:
        # Look at the info
        print(df.info())  # All ints
        # Are there any nulls?
        print(df.isnull().sum())  # None!
        # Are there any duplicate EmployeeIDs?
        print(df["EmployeeID"].is_unique)  # There are none

    # No need to drop nulls or change dtypes
    return df


def load_in_time_data(verbose: bool = False) -> pd.DataFrame:
    """Load in time data from csv file and return dataframe."""

    # Read data from csv file
    df = pd.read_csv("../../Datasets/in_time.csv")

    # Delete first column
    df.drop(df.columns[0], axis=1, inplace=True)

    if verbose:
        # Look at the info
        print(df.info())
        print(df.head())

        # count nulls
        print(df.isnull().sum())  # there are loads of nulls

        # do any columns consist of all nulls?
        print(f"Do any columns consist of all nulls? {df.isnull().all().any()}")  # yes

    # If column consists of lots of nulls, drop it
    df.dropna(axis=1, thresh=1000, inplace=True)

    # Transform string (YYYY-MM-DD hh:mm:ss) values in all columns with datetime
    for col in df.columns:
        df[col] = pd.to_datetime(df[col])

    # Replace Not a Time (NaT) values with average time of column
    for col in df.columns:
        df[col].fillna(df[col].mean(), inplace=True)

    return df


def load_out_time_data(verbose: bool = False) -> pd.DataFrame:
    """Load out time data from csv file and return dataframe."""

    # Read data from csv file
    df = pd.read_csv("../../Datasets/out_time.csv")

    # Delete first column
    df.drop(df.columns[0], axis=1, inplace=True)

    if verbose:
        # Look at the info
        print(df.info())
        print(df.head())

        # count nulls
        print(df.isnull().sum())  # there are loads of nulls

        # do any columns consist of all nulls?
        print(f"Do any columns consist of all nulls? {df.isnull().all().any()}")  # yes

    # If column consists of lots of nulls, drop it
    df.dropna(axis=1, thresh=1000, inplace=True)

    # Transform string (YYYY-MM-DD hh:mm:ss) values in all columns with datetime
    for col in df.columns:
        df[col] = pd.to_datetime(df[col])

    # Replace Not a Time (NaT) values with average time of column
    for col in df.columns:
        df[col].fillna(df[col].mean(), inplace=True)

    return df


if __name__ == "__main__":
    VERBOSE = True

    # Load data
    general_data = load_general_data()
    employee_survey_data = load_employee_survey_data()
    manager_survey_data = load_manager_survey_data()
    in_time_data = load_in_time_data()
    out_time_data = load_out_time_data()

    # Add features to data

    # Create new dataframe with day and length of day

    # Create empty dataframe
    day_length_df = pd.DataFrame()

    for day_start in in_time_data.columns:
        # Check if day_end is in out_time_data
        if day_start in out_time_data.columns:
            # Calculate length of day using mean of in_time and out_time
            day_end = out_time_data.columns[in_time_data.columns.get_loc(day_start)]
        else:
            continue

        # Use concat to avoid PerformanceWarning
        day_length_df = pd.concat(
            [
                day_length_df,
                pd.DataFrame(
                    {
                        f"{day_start}_length": (
                            out_time_data[day_end] - in_time_data[day_start]
                        ).dt.seconds
                        / (60 * 60)
                    }
                ),
            ],
            axis=1,
        )

    # Check if employees are working overtime (e.g., over eight hours a day)
    day_length_df["has_ever_worked_overtime"] = day_length_df.apply(
        lambda row: 1 if row.max() > 8 else 0, axis=1
    )
    # Calculate percentage of days worked overtime
    day_length_df["overtime_percentage"] = day_length_df.apply(
        lambda row: row.sum() / len(row), axis=1
    )

    print(day_length_df.head())
    print(day_length_df.info())
    print(day_length_df.describe())

    # Change employee education number to word
    general_data["EducationLevel"] = general_data["Education"].map(
        {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}
    )

    if VERBOSE:
        print(general_data["EducationLevel"].value_counts())
        print(general_data.info())

    satisfaction_level = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}

    # Change environment satisfaction to word
    employee_survey_data["EnvironmentSatisfactionLevel"] = employee_survey_data[
        "EnvironmentSatisfaction"
    ].map(satisfaction_level)

    # Change job satisfaction to word
    employee_survey_data["JobSatisfactionLevel"] = employee_survey_data[
        "JobSatisfaction"
    ].map(satisfaction_level)

    # Job involvement to word
    manager_survey_data["JobInvolvementLevel"] = manager_survey_data[
        "JobInvolvement"
    ].map(satisfaction_level)

    # Performance rating to word
    manager_survey_data["PerformanceRatingLevel"] = manager_survey_data[
        "PerformanceRating"
    ].map({1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding"})

    # Work-life balance to word
    employee_survey_data["WorkLifeBalanceLevel"] = employee_survey_data[
        "WorkLifeBalance"
    ].map({1: "Bad", 2: "Good", 3: "Better", 4: "Best"})

    if VERBOSE:
        # Check descriptive stats
        print(employee_survey_data.describe())
        print(manager_survey_data.describe())
        print(general_data.describe())
        # Looks good!

    # Remove underscore from travel frequency
    general_data["BusinessTravel"] = general_data["BusinessTravel"].str.replace(
        "_", " "
    )

    # categorise incomes into income bands
    lower_quartile_income = general_data["MonthlyIncome"].quantile(0.25)
    upper_quartile_income = general_data["MonthlyIncome"].quantile(0.75)
    mean_income = general_data["MonthlyIncome"].mean()

    # create income bands
    general_data["IncomeBand"] = pd.cut(
        general_data["MonthlyIncome"],
        bins=[0, lower_quartile_income, mean_income, upper_quartile_income, np.inf],
        labels=["Low", "Medium", "High", "Very High"],
    )

    if VERBOSE:
        # describe IncomeBand
        print(general_data["IncomeBand"].describe())
