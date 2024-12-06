import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from missforest import MissForest
from lazypredict.Supervised import LazyRegressor
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

data = pd.read_csv("CustomerChurn.csv")
target = "Churn"

# profile = ProfileReport(data, title="Customer Churn Report", explorative=True)
# profile.to_file("Churn.html")

data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
# print(data.info())

# print(data[["Age", "Tenure", "MonthlyCharges", "TotalCharges"]].corr())

# X = data[["Tenure", "TotalCharges"]]  
# X = X.assign(intercept=1)  
# vif = pd.DataFrame()
# vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# vif["Feature"] = X.columns
# print(vif) 

#VIF both > 5, multicolinearity, need to drop 1 of them, I choose TotalCharges

data = data.drop(columns=["TotalCharges"])

x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10091991)

numerical_features = ["Age", "Tenure", "MonthlyCharges"]
ord_features = ["ContractType", "TechSupport", "Gender"]
nom_features = ["InternetService"]

# Impute missing values using MissForest
miss_forest = MissForest()
x_train_num = x_train[numerical_features]
x_test_num = x_test[numerical_features]
if x_train_num.isnull().values.any():
    miss_forest = MissForest()
    x_train_num_imputed = miss_forest.fit_transform(x_train_num)
    x_test_num_imputed = miss_forest.transform(x_test_num)
else:
    x_train_num_imputed = x_train_num.to_numpy()
    x_test_num_imputed = x_test_num.to_numpy()


num_transformer = Pipeline(steps=[
    ("scaler", StandardScaler()),
])

ContractType_order = ['Month-to-Month', 'One-Year', 'Two-Year']
Gender_order = data["Gender"].unique()
TechSupport_order = data["TechSupport"].unique()

ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[
        ContractType_order,  
        TechSupport_order,   
        Gender_order        
    ])),
])

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num_features", num_transformer, numerical_features),
    ("ord_features", ord_transformer, ord_features),
    ("nom_features", nom_transformer, nom_features),
])

if y_train.dtypes == 'object':
    y_train = y_train.astype('category').cat.codes
    y_test = y_test.astype('category').cat.codes

# x_train_processed = preprocessor.fit_transform(x_train)
# x_test_processed = preprocessor.transform(x_test)

# regressor = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = regressor.fit(x_train_processed, x_test_processed, y_train, y_test)

# print(models)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=100))
])

model.fit(x_train, y_train)
y_predict = model.predict(x_test)

params = {
    "regressor__n_estimators": [50, 100, 200],
    "regressor__criterion": ["squared_error", "absolute_error", "poisson"],
    "regressor__max_depth": [None, 2, 5],
    "regressor__min_samples_split": [2, 5, 10],
}

model = GridSearchCV(model, param_grid=params, scoring="r2", cv=6, verbose=2, n_jobs=6)

model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print(model.best_params_)
print(model.best_score_)

print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
print("R2: {}".format(r2_score(y_test, y_predict)))

#Without GridSearch: MAE: 0.0068, MSE: 0.001932, R2: 0.9817045454545454

x_train_processed = preprocessor.fit_transform(x_train)
x_test_processed = preprocessor.transform(x_test)

kmeans = KMeans(n_clusters=3, random_state=100)
clusters = kmeans.fit_predict(x_train_processed)

#I don't know why ChatGPT need to change x_train_processed into dataframe
x_train_processed_df = pd.DataFrame(
    x_train_processed, 
    columns=preprocessor.get_feature_names_out(input_features=x_train.columns)
)

x_train_processed_df["Cluster"] = clusters

print("K-Means Cluster Centers:\n", kmeans.cluster_centers_)


