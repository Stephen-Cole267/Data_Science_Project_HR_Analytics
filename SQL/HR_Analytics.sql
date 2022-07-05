/* THIS SQL SCRIPT CREATES THE TABLES NEEDED FOR THE DA AND DS BENCH PROJECTS */
/* CREATION DATE: 2022-07-05 */
/* CREATED BY: STEPHEN COLE */

/* LAST UPDATED: 2022-07-05 */
/* LAST UPDATED BY: STEPHEN COLE */

/* FOR BRIEF PLEASE SEE THE FOLLOWING LINK: https://github.com/Stephen-Cole267/Data_Science_Project_HR_Analytics */

/***************** CREATING employee_survey_data TABLE *****************/
drop table if exists employee_survey_data;

create table employee_survey_data (
    "EmployeeID" BIGINT PRIMARY KEY NOT NULL,
    "EnvironmentSatisfaction" INT,
    "JobSatisfaction" INT,
    "WorkLifeBalance" INT
);

copy employee_survey_data from '[PATH_TO_CSV]' NULL as 'NA' delimiter ',' CSV HEADER;  ----- CHANGE FILE PATH TO employee_survey_data.csv HERE


/***************** CREATING general_data TABLE *****************/

drop table if exists general_data;

create table general_data (
    "Age" INT,
    "Attrition" VARCHAR(3),
    "BusinessTravel" VARCHAR(355),
    "Department" VARCHAR(355),
    "DistanceFromHome" INT,
    "Education" INT,
    "EducationField" VARCHAR(355),
    "EmployeeCount" BIGINT,
    "EmployeeID" BIGINT PRIMARY KEY NOT NULL,
    "Gender" varchar(355),
    "JobLevel" INT,
    "JobRole" VARCHAR(355),
    "MaritalStatus" VARCHAR(355),
    "MonthlyIncome" BIGINT,
    "NumCompaniesWorked" INT,
    "Over18" VARCHAR(1),
    "PercentSalaryHike" INT,
    "StandardHours" INT,
    "StockOptionLevel" INT,
    "TotalWorkingYears" INT,
    "TrainingTimesLastYear" INT,
    "YearsAtCompany" INT,
    "YearsSinceLastPromotion" INT,
    "YearsWithCurrManager" INT
);

copy general_data from '[PATH_TO_CSV]' NULL as 'NA' delimiter ',' CSV HEADER; ----- CHANGE FILE PATH TO general_data.csv HERE


/***************** CREATING manager_survey TABLE *****************/

drop table if exists manager_survey_data;

create table manager_survey_data (
    "EmployeeID" BIGINT PRIMARY KEY NOT NULL,
    "JobInvolvement" INT,
    "PerformanceRating" INT
);

copy manager_survey_data from '[PATH_TO_CSV]' NULL as 'NA' delimiter ',' CSV HEADER; ----- CHANGE FILE PATH TO manager_survey_data.csv HERE

/* Verify that all tables have been copied successfully by running the below */

CREATE TEMP TABLE TABLE_ROW_COUNT AS
	(SELECT 'employee_survey_data' AS TABLE_NAME,
			COUNT(*) AS NO_ROWS
		FROM EMPLOYEE_SURVEY_DATA
		UNION ALL SELECT 'general_data' AS TABLE_NAME,
			COUNT(*) AS NO_ROWS
		FROM GENERAL_DATA
		UNION ALL SELECT 'manager_survey_data' AS TABLE_NAME,
			COUNT(*) AS NO_ROWS
		FROM MANAGER_SURVEY_DATA);

CREATE TEMP TABLE TABLE_COLUMN_COUNT AS
	(SELECT TABLE_NAME,
			COUNT(COLUMN_NAME) AS NO_COLUMNS
		FROM INFORMATION_SCHEMA.COLUMNS
		WHERE TABLE_NAME = 'employee_survey_data'
			OR TABLE_NAME = 'general_data'
			OR TABLE_NAME = 'manager_survey_data'
		GROUP BY 1);

CREATE TABLE TABLE_MART AS
	(SELECT A.*,
			B.NO_ROWS
		FROM TABLE_COLUMN_COUNT AS A
		LEFT JOIN TABLE_ROW_COUNT AS B ON A.TABLE_NAME = B.TABLE_NAME);

/*************************************************** 
Verify that: 
- all tables have 4410 rows 
- employee_survey_data has 4 columns
- general_data has 24 columns
- manager_survey_data has 3 columns
***************************************************/

SELECT * FROM TABLE_MART;
