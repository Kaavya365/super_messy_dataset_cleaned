# importing relevant libraries
import pandas as pd
import numpy as np

# reading the raw data
df = pd.read_csv('C:/Users/kaavy/downloads/DP_testprep/scholarship-cleaning/data/raw/scholarship_applications_raw.csv')

# copying the raw data to a new dataframe for cleaning
raw = df.copy()

# displaying the first few rows of the raw data for inspection
raw.head()

# cleaning the column names using the following pipeline: strip whitespace, convert to lowercase as standardisation, replace spaces with underscores, remove special characters, and remove trailing underscores
raw.columns = (
    raw.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace(r"[^a-z0-9_]", "", regex=True)
    .str.rstrip('_')
)

# dropping columns that contain all null values 
raw.dropna(axis=1, how='all', inplace=True)

# removing duplicate rows while keeping the first occurrence
raw.drop_duplicates(inplace=True, keep='first')

# removing leading and trailing whitespace from the 'student_name' column and standardizing the names to title case, replacing multiple spaces with a single space, and removing any non-alphabetic characters
raw['student_name'] = raw['student_name'].str.strip().str.title().str.replace(r'\s+', ' ', regex=True).str.replace(r'[^a-zA-Z\s]', '', regex=True)

# cleaning the 'state' column by stripping whitespace, converting to title case, and mapping various representations of state names to a standardized format using a dictionary
state_map = {
    'U.P.': 'Uttar Pradesh',
    "Mh": 'Maharashtra',
    "W. Bengal": 'West Bengal',
    "Bihar": 'Bihar',
    "Kerala": 'Kerala',
    "Uttar Pradesh": 'Uttar Pradesh',
    "West Bengal": 'West Bengal',
    "Delhi": 'Delhi',
    "Tamil Nadu": 'Tamil Nadu',
    "nan": np.nan,
    "Wb": 'West Bengal',
    "Maharashtra": 'Maharashtra',
    "Kerela": 'Kerala',
    "Maharastra": 'Maharashtra',
    "Tamil  Nadu": 'Tamil Nadu',
    "Up": 'Uttar Pradesh',
    "New Delhi": 'New Delhi',
    "Tamilnadu": 'Tamil Nadu',
    "Nct Delhi": 'New Delhi',
    "Tn": 'Tamil Nadu',
    "Uttar Pardesh": 'Uttar Pradesh',
}
raw['state'] = raw['state'].map(state_map)

# cleaning the 'gender' column by stripping whitespace, converting to title case, and mapping various representations of gender to a standardized format using a dictionary
gender_map = {
    'Male': 'Male',
    'male': 'Male',
    'f': 'Female',
    'Female': 'Female',
    'MALE': 'Male',
    ' F ': 'Female',
    'm': 'Male',
    'Prefer not to say': 'Prefer not to say',
    'female': 'Female',
    'nan': np.nan,
    'FEMALE': 'Female',
    'M ': 'Male',
    'M': 'Male',
    'F': 'Female',
}
raw['gender'] = raw['gender'].map(gender_map)

# cleaning the 'category' column by stripping whitespace, converting to title case, and mapping various representations of category to a standardized format using a dictionary
category_map = {
    'ST': 'Scheduled Tribe',
    'general': 'General',
    'O.B.C': 'Other Backward Class',
    'obc': 'Other Backward Class',
    'General': 'General',
    'nan': np.nan,
    'GEN': 'General',
    'OBC': 'Other Backward Class',
    'S.C.': 'Scheduled Caste',
    'SC': 'Scheduled Caste',
    'Gen': 'General',
    'st': 'Scheduled Tribe',
    'sc': 'Scheduled Caste',
}
raw['category'] = raw['category'].map(category_map)

# cleaning the 'hostel_required' column by mapping various representations of hostel requirement to a standardized format using a dictionary
hostel_map = {
    'FALSE': 'No',
    'TRUE': 'Yes',
    'no': 'No',
    'yes': 'Yes',
    'Y': 'Yes',
    'N': 'No',
    '0': 'No',
    '1': 'Yes',
    'No': 'No',
    'Yes': 'Yes'
}
raw['hostel_required'] = raw['hostel_required'].map(hostel_map)

# cleaning the 'family_income' column by removing currency symbols, commas, and other non-numeric characters, converting to numeric, and flagging negative and undisclosed values
raw['family_income'] = (
    raw['family_income']
    .str.strip()
    .str.replace('Rs.', '', regex=False)
    .str.replace(',', '', regex=False)
    .str.replace('INR', '', regex=False)
    .str.replace('NAN', '', regex=False)
    .str.replace('lakh', '00000', regex=False)
    .str.replace('0.', '', regex=False)
)
raw['negative_family_income'] = raw['family_income'].str.contains('-', na=False)
raw['family_income'] = raw['family_income'].str.replace('-', '', regex=False)
raw['undisclosed_family_income'] = (
    raw['family_income'].str.contains('not disclosed', na=False) | 
    raw['family_income'].str.contains('NA', na=False) | 
    raw['family_income'].str.contains('Not Available', na=False) | 
    raw['family_income'].str.contains('Not Disclosed', na=False) |
    raw['family_income'].str.contains('', na=False) 
)
raw['family_income'] = (
    raw['family_income']
    .str.replace('not disclosed', '', regex=False)
    .str.replace('NA', '', regex=False)
    .str.replace('Not Available', '', regex=False)
    .str.replace('Not Disclosed', '', regex=False)
)
raw['family_income'] = pd.to_numeric(raw['family_income'], errors='coerce')

# cleaning the 'marks' column by converting to numeric and flagging invalid values
raw['marks'] = pd.to_numeric(raw['marks'], errors='coerce')
raw['marks_was_invalid'] = (
    (raw['marks'] > 100) |
    (raw['marks'] < 0) |
    (raw['marks'].isna())
)
raw['marks'] = (
    raw['marks']
    .where((raw['marks'] <= 100) & (raw['marks'] >= 0), np.nan)
)

# cleaning the 'attendance' column by removing the percentage sign, converting to numeric, and flagging invalid values
raw['attendance'] = (
    raw['attendance']
    .str.replace('%', '', regex=False)
)
raw['attendance'] = pd.to_numeric(raw['attendance'], errors='coerce')
raw['attendance_was_invalid'] = (
    (raw['attendance'] > 100) |
    (raw['attendance'] < 0) |
    (raw['attendance'].isna())
)
raw['attendance'] = (
    raw['attendance']
    .where((raw['attendance'] <= 100) & (raw['attendance'] >= 0), np.nan)
)

# filling missing values in the 'marks' and 'attendance' columns with the median of their respective columns and creating flags for missing values
raw['marks_was_missing'] = raw['marks'].isna()
raw['attendance_was_missing'] = raw['attendance'].isna()
raw['marks'] = raw['marks'].fillna(raw['marks'].median())
raw['attendance'] = raw['attendance'].fillna(raw['attendance'].median())

# cleaning the 'application_date' column by converting to datetime format, flagging invalid dates, and handling errors
raw['application_date'] = pd.to_datetime(raw['application_date'], format='mixed', dayfirst=True, errors='coerce')
raw['application_date_was_invalid'] = (raw['application_date'].isna() | (raw['application_date'] > pd.Timestamp.now()))

# dropping rows with missing 'application_id' and removing duplicate 'application_id' entries while keeping the first occurrence
raw = raw.dropna(subset=['application_id'])
raw = raw.drop_duplicates(
    subset=['application_id'],
    keep='first'
)

# converting 'application_id' to string and removing any trailing '.0' that came from the numeric dt
raw['application_id'] = raw['application_id'].astype('string').str.replace('.0', '', regex=False)

# creating an income outlier flag
q1 = raw['family_income'].quantile(0.25)
q2 = raw['family_income'].quantile(0.75)
iqr = q2 - q1
lower = q1 - 1.5 * iqr
upper = q2 + 1.5 * iqr
possible_outliers = raw[(raw['family_income'] < lower) | (raw['family_income'] > upper)]
raw['family_income_outlier'] = raw['application_id'].isin(possible_outliers['application_id'])
print(possible_outliers)

# creating a results feature based on marks:
raw['results'] = np.select([raw['marks'] >= 60, raw['marks'] < 60, raw['marks'].isna()], ['Pass', 'Fail', 'Missing'], default='Missing')
print(raw['results'])

raw['marks'] = raw['marks'].astype('Int64')
raw['state'] = raw['state'].astype('category')
raw['gender'] = raw['gender'].astype('category')
raw['category'] = raw['category'].astype('category')
raw['hostel_required'] = raw['hostel_required'].astype('bool')
raw['application_date'] = raw['application_date'].astype('datetime64[ns]')

raw.to_csv('data/clean/scholarship_clean.csv', index=False)
