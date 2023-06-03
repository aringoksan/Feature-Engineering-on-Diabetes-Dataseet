from sklearn.preprocessing import StandardScaler,MinMaxScaler, RobustScaler
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 5000)
diabetes = pd.read_csv("diabetes.csv")
outcome = diabetes["Outcome"]
properties = diabetes.drop("Outcome", axis=1)


properties.info()
properties.shape
cat_cols = [col for col in properties.columns if properties[col].dtype == "O"]
num_cols = [col for col in properties.columns if properties[col].dtypes in ["int64", "float"]]
properties.describe().T


for col in diabetes.columns:
    print(diabetes.groupby(col).agg({"Outcome": "mean"}))

def quantile_ranges(dataframe, classical=True):
    upper_limit = []
    lower_limit = []
    if classical:
        upper_limit_ratio = 0.75
        lower_limit_ratio = 0.25
    else:
        upper_limit_ratio = float(input("What is upper limit: "))
        lower_limit_ratio = float(input("What is lower limit: "))
    for col in dataframe.columns:
            upper_quantiles = dataframe[col].quantile(upper_limit_ratio)
            lower_quantiles = dataframe[col].quantile(lower_limit_ratio)
            interquantile_range = dataframe[col].quantile(upper_limit_ratio) - dataframe[col].quantile(lower_limit_ratio)
            upper_limit.append(upper_quantiles + 1.5 * interquantile_range)
            lower_limit.append(lower_quantiles - 1.5 * interquantile_range)
    return upper_limit, lower_limit
def outlier_supression(dataframe, u_lim, l_lim, drop_outliers=False):
    if drop_outliers:
        for i, col in enumerate(dataframe.columns):
            dataframe = dataframe.loc[(dataframe[col] < u_lim[i]) & (dataframe[col] > l_lim[i])]
        print(dataframe.describe().T)
    else:
        for i, col in enumerate(dataframe.columns):
            dataframe.loc[(dataframe[col] > u_lim[i]), col] = u_lim[i]
            dataframe.loc[(dataframe[col] < l_lim[i]), col] = l_lim[i]
        print(dataframe.describe().T)
    return dataframe

upper_lim, lower_lim = quantile_ranges(properties)
properties = outlier_supression(properties, upper_lim, lower_lim)
print(properties.isnull().sum())
print(outcome.isnull().sum())
for col in properties.columns:
    print(f"Columns: {col}, Count: {properties.loc[properties[col] == 0, col].count()}")

# properties.replace(to_replace=0, value=np.NaN, inplace=True)
# properties.isna().sum()
# for col in properties.columns:
#    properties[col].fillna(properties.loc[properties[col].notnull(), col].mean(), inplace=True)

properties.loc[properties["SkinThickness"] == 0, "SkinThickness"] = properties.loc[properties["SkinThickness"] != 0, "SkinThickness"].mean()
properties.loc[properties["Insulin"] == 0, "Insulin"] = properties.loc[properties["Insulin"] != 0, "Insulin"].mean()
properties.loc[(properties["SkinThickness"] == 0) | (properties["Insulin"] == 0)]
properties["AGE_BMI"] = properties["Age"] + properties["BMI"]
properties["Glucose_Age_Ratio"] = properties["Glucose"] + properties["Age"]

# properties.replace(to_replace=properties.loc[properties["SkinThickness"] == 0, "SkinThickness"],
#                   value=properties.loc[properties["SkinThickness"] != 0, "SkinThickness"].mean())
corr_dataframe = properties.corrwith(outcome, axis=0)

def standardization(dataframe, standardization_method=0):
    df = dataframe.copy()
    if standardization_method == 0:
        scalar = StandardScaler()
        for col in df.columns:
            df[col] = scalar.fit_transform(df[[col]])
    elif standardization_method == 1:
        scalar = MinMaxScaler()
        for col in df.columns:
            df[col] = scalar.fit_transform(df[[col]])
    else:
        scalar = RobustScaler()
        for col in df.columns:
            df[col] = scalar.fit_transform(df[[col]])
    return df


minmaxscaled_properties = standardization(properties, standardization_method=1)
standartscaled_properties = standardization(properties, standardization_method=0)
robustscaled_properties = standardization(properties, standardization_method=2)

X_train, X_test, Y_train, Y_test = train_test_split(minmaxscaled_properties, outcome, test_size=0.25)
rf_model = RandomForestClassifier().fit(X_train, Y_train)
y_prediction = rf_model.predict(X_test)
accuracy_score(y_prediction, Y_test)
