import pandas as pd


def load_general_data(verbose=False):
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


def load_out_time_data():
    # Read data from csv file
    df = pd.read_csv("../../Datasets/out_time.csv")

    # TODO: what is this?
    print(df.info())
    print(df.head())
    return df


def load_employee_survey_data(verbose=False):
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


def load_manager_survey_data(verbose=False):
    # Read data from csv file
    df = pd.read_csv("../../Datasets/manager_survey_data.csv")

    if verbose:
        # Look at the info
        print(df.info())  # All ints

        # Are there any nulls?
        print(df.isnull().sum())  # None!

        # Are there any duplicate EmployeeIDs?
        print(df["EmployeeID"].is_unique)  # True

    # No need to drop nulls or change dtypes
    return df


def load_in_time_data(verbose=False):
    # Read data from csv file
    df = pd.read_csv("../../Datasets/in_time.csv")

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


def load_out_time_data(verbose=False):
    # Read data from csv file
    df = pd.read_csv("../../Datasets/out_time.csv")

    if verbose:
        # Look at the info
        print(df.info())
        print(df.head())

        # count nulls
        print(df.isnull().sum()) # there are loads of nulls

        # do any columns consist of all nulls?
        print(f"Do any columns consist of all nulls? {df.isnull().all().any()}") # yes

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
    verbose = True

    # Load data
    general_data = load_general_data()
    employee_survey_data = load_employee_survey_data()
    manager_survey_data = load_manager_survey_data()
    in_time_data = load_in_time_data()
    out_time_data = load_out_time_data()

    # Add features to data

    # Change employee education number to word
    general_data["EducationLevel"] = general_data["Education"].map(
        {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}
    )

    if verbose:
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

    # Relationship satisfaction to word
    # employee_survey_data["RelationshipSatisfactionLevel"] = employee_survey_data["RelationshipSatisfaction"].map(satisfaction_level)
    # TODO: Ask why this is missing?

    # Performance rating to word
    manager_survey_data["PerformanceRatingLevel"] = manager_survey_data[
        "PerformanceRating"
    ].map({1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding"})

    # Work life balance to word
    employee_survey_data["WorkLifeBalanceLevel"] = employee_survey_data[
        "WorkLifeBalance"
    ].map({1: "Bad", 2: "Good", 3: "Better", 4: "Best"})

    if verbose:
        # Check descriptive stats
        print(employee_survey_data.describe())
        print(manager_survey_data.describe())
        print(general_data.describe())
        # Looks good!

    # Remove underscore from travel frequency
    general_data["BusinessTravel"] = general_data["BusinessTravel"].str.replace(
        "_", " "
    )

    # todo: what is stock option level?
