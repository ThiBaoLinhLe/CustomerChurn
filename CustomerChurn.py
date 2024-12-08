import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, accuracy_score, f1_score
from missforest import MissForest
from lazypredict.Supervised import LazyClassifier
from sklearn.cluster import KMeans
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

# regressor = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = regressor.fit(x_train_processed, x_test_processed, y_train, y_test)

# print(models)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=100))
])

model.fit(x_train, y_train)
y_predict = model.predict(x_test)

param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__criterion": ["gini", "entropy", "log_loss"],
    "classifier__max_depth": [None, 5, 10],
    "classifier__min_samples_split": [2, 5, 10],
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="accuracy",
    cv=6,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_

y_pred = best_model.predict(x_test)
y_prob = best_model.predict_proba(x_test)[:, 1]

print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label="ROC Curve (AUC = {:.2f})".format(roc_auc_score(y_test, y_prob)))
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.show()


x_train_processed = preprocessor.fit_transform(x_train)
x_test_processed = preprocessor.transform(x_test)

kmeans = KMeans(n_clusters=3, random_state=100)
clusters = kmeans.fit_predict(x_train_processed)

# #I don't know why ChatGPT need to change x_train_processed into dataframe
x_train_processed_df = pd.DataFrame(
    x_train_processed, 
    columns=preprocessor.get_feature_names_out(input_features=x_train.columns)
)

x_train_processed_df["Cluster"] = clusters

print("K-Means Cluster Centers:\n", kmeans.cluster_centers_)


