# HR Analytics DA/DS Project
> NOTE: This is a Data Analytics/Science project for those who want to test their industry skills. Please upload your solutions by opening a new branch to the main. All files that are instantly merged to the main will be deleted.

## Project Brief
A large company named JWC, employs, at any given point of time, around 4000 employees. However, every year, around some of its employees leave the company, for various reasons, and need to be replaced with the talent pool available in the job market. The management believes that this level of attrition (employees leaving, either on their own or because they got fired) is bad for the company, because of the following reasons:

1. The former employees’ projects get delayed, which makes it difficult to meet timelines, resulting in a reputation loss among consumers and partners.
2. A sizeable department has to be maintained, for the purposes of recruiting new talent.
3. More often than not, the new employees have to be trained for the job and/or given time to acclimatise themselves to the company.

Hence, the management has contracted an HR analytics firm, Xander Talent HR, to understand what JWC can do to stop minimise attrition as much as possible. They want to know what changes they should make to their workplace, in order to get most of their employees to stay. Also, they want to know which of these variables is most important and needs to be addressed right away.

##### Goal of the case study

> Data Analyst - You are required to report the attributes of attrition and what other patterns you can recognise related attrition. The results thus obtained will be used by the management to understand what changes they should make to their workplace, in order to get most of their employees to stay.

> Data Science - You are required to model the probability of attrition using a logistic regression. The results thus obtained will be used by the management to understand what changes they should make to their workplace, in order to get most of their employees to stay.

For this project, Stephen Cole & Alex Naylor will be the stakeholders.

## Getting Started
The data will ideally be in a cloud database which uses some form of SQL, i.e. AWS with MySQL db, Azure with SQL Server, etc. Therefore, we will want to get the data into postgreSQL so that we can do any data gathering for our objective. To get the database set up:

1. Download the CSVs within the [Datasets](https://github.com/Stephen-Cole267/Data_Science_Project_HR_Analytics/tree/main/Datasets) folder into your directory
> Note: Keep track of where you save these CSVs as you will need to add some of the paths to the SQL file.
2. Create a database in pgAdmin 4 (Can call it whatever you want)
3. Create tables within your newly created database using the [SQL](https://github.com/Stephen-Cole267/Data_Science_Project_HR_Analytics/blob/main/SQL/HR_Analytics.sql) file
> Note: You need to change the `[PATH_TO_CSV]` to the relevant csv files. The tables are named after their CSV counterparts.

 If you get a permission error then you will need to change the permissions of your folder that has the csvs. This can be done by going right clicking on your folder and clicking on `Properties` --> `Security` --> `Edit` --> `Add..` then typing in "Everyone" and pressing `Ok`. This will give pgadmin 4 access to whatever you put in that folder. You can verify this by checking that there is a tick next to `Read & execute` in the properties of your folder.

You are now ready to start the task. If you have any issues please post within the Data Analytics/Data Science Forum.

The data dictionary for the tables within SQL can be found [here](https://github.com/Stephen-Cole267/Data_Science_Project_HR_Analytics/blob/main/Datasets/data_dictionary.xlsx).

## Tips

### Requirements Gathering
The start to any project is to make sure you have clear and well-defined requirements for your project. Most projects start with a vague idea of what the stakeholder wants, and as a consultant, we will never have as much knowledge about their problem/business context as they do. Therefore, we need to get as much information out of them as possible, as they will subconsciously assume that we know everything. For this project, Stephen Cole & Alex Naylor will be the stakeholders.

> If you don't know the answer to any question then you should always ask - NEVER ASSUME. This will only risk the accuracy of your work and end up having to do everything all over again if you wrongly assume.

Questions to ask yourself constantly throughout the project are:
- What is the purpose of this project, why does the stakeholder want this and what is the desired outcome of the project?
- Is there any extra info that the stakeholder could tell you to help tailor the project to what they want?

Questions more specifically for Data Science:
- Are there any features that I could create to improve the model?
- What machine learning models are applicable to the project's target?
- Is the target feature for my model correct?

If you do not know the answer or are unsure, then I highly recommend asking the stakeholders.

## Results
### Data Analytics
A report of your findings either through a presentation or a dashboard or even both!

### Data Science
A machine learning model that predicts the target, with a report of the model's performance and any other analytics.


