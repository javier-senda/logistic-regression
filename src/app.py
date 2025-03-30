import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif, SelectKBest
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

# Importar dataset
total_data=pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv", sep=';')

total_data.to_csv("../data/raw/total_data.csv", index = False)

# Eliminar duplicados
total_data = total_data.drop_duplicates().reset_index(drop = True)


# Análisis de variables univariante

## Categóricas
fig, axis = plt.subplots(3, 4, figsize = (24, 10))

sns.histplot(ax = axis[0, 0], data = total_data, x = "age")
sns.histplot(ax = axis[0, 1], data = total_data, x = "job")
sns.histplot(ax = axis[0, 2], data = total_data, x = "marital")
sns.histplot(ax = axis[0, 3], data = total_data, x = "education")
sns.histplot(ax = axis[1, 0], data = total_data, x = "default")
sns.histplot(ax = axis[1, 1], data = total_data, x = "housing")
sns.histplot(ax = axis[1, 2], data = total_data, x = "loan")
sns.histplot(ax = axis[1, 3], data = total_data, x = "contact")
sns.histplot(ax = axis[2, 0], data = total_data, x = "month")
sns.histplot(ax = axis[2, 1], data = total_data, x = "day_of_week")
sns.histplot(ax = axis[2, 2], data = total_data, x = "poutcome")
sns.histplot(ax = axis[2, 3], data = total_data, x = "y")



for row in axis:
    for ax in row:
        if ax in fig.axes:
            ax.tick_params(axis='x', rotation=45)

plt.tight_layout()

plt.show()

## Numéricas
fig, axis = plt.subplots(4, 5, figsize=(18, 8), gridspec_kw={"height_ratios": [6, 1] * 2})

sns.histplot(ax=axis[0, 0], data=total_data, x="duration")
sns.boxplot(ax=axis[1, 0], data=total_data, x="duration")

sns.histplot(ax=axis[0, 1], data=total_data, x="campaign")
sns.boxplot(ax=axis[1, 1], data=total_data, x="campaign")

sns.histplot(ax=axis[0, 2], data=total_data, x="pdays")
sns.boxplot(ax=axis[1, 2], data=total_data, x="pdays")

sns.histplot(ax=axis[0, 3], data=total_data, x="previous")
sns.boxplot(ax=axis[1, 3], data=total_data, x="previous")

sns.histplot(ax=axis[0, 4], data=total_data, x="emp.var.rate")
sns.boxplot(ax=axis[1, 4], data=total_data, x="emp.var.rate")

sns.histplot(ax=axis[2, 0], data=total_data, x="cons.price.idx")
sns.boxplot(ax=axis[3, 0], data=total_data, x="cons.price.idx")

sns.histplot(ax=axis[2, 1], data=total_data, x="cons.conf.idx")
sns.boxplot(ax=axis[3, 1], data=total_data, x="cons.conf.idx")

sns.histplot(ax=axis[2, 2], data=total_data, x="euribor3m")
sns.boxplot(ax=axis[3, 2], data=total_data, x="euribor3m")

sns.histplot(ax=axis[2, 3], data=total_data, x="nr.employed")
sns.boxplot(ax=axis[3, 3], data=total_data, x="nr.employed")

fig.delaxes(axis[2,4])
fig.delaxes(axis[3,4])

plt.tight_layout()

plt.show()

total_data["was_contacted"] = total_data["pdays"] != 999

total_data = total_data.drop(columns=["pdays"])

# Análisis de variables multivariante

## Categórico - Categórico
categoricas_normalizadas = [
    "job_n", "marital_n", "education_n", "default_n",
    "housing_n", "loan_n", "contact_n", "month_n",
    "day_of_week_n", "was_contacted", "y_n"
]


total_data["job_n"] = pd.factorize(total_data["job"])[0]
total_data["marital_n"] = pd.factorize(total_data["marital"])[0]
total_data["education_n"] = pd.factorize(total_data["education"])[0]
total_data["default_n"] = pd.factorize(total_data["default"])[0]
total_data["housing_n"] = pd.factorize(total_data["housing"])[0]
total_data["loan_n"] = pd.factorize(total_data["loan"])[0]
total_data["contact_n"] = pd.factorize(total_data["contact"])[0]
total_data["month_n"] = pd.factorize(total_data["month"])[0]
total_data["day_of_week_n"] = pd.factorize(total_data["day_of_week"])[0]
total_data["was_contacted_n"] = pd.factorize(total_data["was_contacted"])[0]
total_data["y_n"] = pd.factorize(total_data["y"])[0]

fig, axis = plt.subplots(figsize = (10, 6))

sns.heatmap(total_data[categoricas_normalizadas].corr(), annot = True, fmt = ".2f")

plt.tight_layout()

plt.show()

columnas = [
    ("job", "job_n"),
    ("marital", "marital_n"),
    ("education", "education_n"),
    ("default", "default_n"),
    ("housing", "housing_n"),
    ("loan", "loan_n"),
    ("contact", "contact_n"),
    ("month", "month_n"),
    ("day_of_week", "day_of_week_n"),
    ("was_contacted", "was_contacted_n"),
    ("y", "y_n")
]

transformation_rules = {}

for original_col, normalized_col in columnas:
    mapping = {
        row[original_col]: row[normalized_col]
        for _, row in total_data[[original_col, normalized_col]].drop_duplicates().iterrows()
    }
    transformation_rules[original_col] = mapping


with open("../models/transformation_rules.json", "w") as f:
    json.dump(transformation_rules, f, indent=4)

## Numérico - categórico
numericas_continuas = [
    "duration", "campaign", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed"
]
categoricas_normalizadas = [
    "job_n", "marital_n", "education_n", "default_n",
    "housing_n", "loan_n", "contact_n", "month_n",
    "day_of_week_n", "was_contacted","y_n"
]

columnas_para_heatmap = numericas_continuas + categoricas_normalizadas

fig, axes = plt.subplots(figsize=(16, 10))

sns.heatmap(total_data[columnas_para_heatmap].corr(), annot = True, fmt = ".2f")

plt.tight_layout()

plt.show()

total_data = total_data.drop(columns=["euribor3m", "nr.employed"])
numericas_continuas.remove("euribor3m")
numericas_continuas.remove("nr.employed")

# Ingeniería de características

total_data_con_outliers = total_data.copy()
total_data_sin_outliers = total_data.copy()

def replace_outliers_from_column(column, df):
  column_stats = df[column].describe()
  column_iqr = column_stats["75%"] - column_stats["25%"]
  upper_limit = column_stats["75%"] + 1.5 * column_iqr
  lower_limit = column_stats["25%"] - 1.5 * column_iqr

  if lower_limit < 0:
    lower_limit = float(df[column].min())
  # Remove upper outliers
  df[column] = df[column].apply(lambda x: x if (x <= upper_limit) else upper_limit)
  # Remove lower outliers
  df[column] = df[column].apply(lambda x: x if (x >= lower_limit) else lower_limit)
  return df.copy(), [lower_limit, upper_limit]

outliers_dict = {}
for column in numericas_continuas:
  total_data_sin_outliers, limits_list = replace_outliers_from_column(column, total_data_sin_outliers)
  outliers_dict[column] = limits_list

with open("../models/outliers_replacement.json", "w") as f:
    json.dump(outliers_dict, f)

# Escalado de valores
num_variables=["age", "duration", "campaign", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", 
               "job_n", "marital_n","education_n", "default_n", "housing_n", "loan_n", "contact_n", "month_n", "day_of_week_n", "was_contacted_n"]

X_con_outliers = total_data_con_outliers.drop("y", axis = 1)[num_variables]
X_sin_outliers = total_data_sin_outliers.drop("y", axis = 1)[num_variables]
y = total_data_con_outliers["y_n"]

X_train_con_outliers, X_test_con_outliers, y_train, y_test = train_test_split(X_con_outliers, y, test_size = 0.2, random_state = 42)
X_train_sin_outliers, X_test_sin_outliers = train_test_split(X_sin_outliers, test_size = 0.2, random_state = 42)


X_train_con_outliers.to_excel("../data/processed/X_train_con_outliers.xlsx", index = False)
X_train_sin_outliers.to_excel("../data/processed/X_train_sin_outliers.xlsx", index = False)
X_test_con_outliers.to_excel("../data/processed/X_test_con_outliers.xlsx", index = False)
X_test_sin_outliers.to_excel("../data/processed/X_test_sin_outliers.xlsx", index = False)
y_train.to_excel("../data/processed/y_train.xlsx", index = False)
y_test.to_excel("../data/processed/y_test.xlsx", index = False)

## Normalización
normalizador_con_outliers = StandardScaler()
normalizador_con_outliers.fit(X_train_con_outliers)

with open("../models/normalizador_con_outliers.pkl", "wb") as file:
    pickle.dump(normalizador_con_outliers,file)

X_train_con_outliers_norm = normalizador_con_outliers.transform(X_train_con_outliers)
X_train_con_outliers_norm = pd.DataFrame(X_train_con_outliers_norm, index = X_train_con_outliers.index, columns = num_variables)

X_test_con_outliers_norm = normalizador_con_outliers.transform(X_test_con_outliers)
X_test_con_outliers_norm = pd.DataFrame(X_test_con_outliers_norm, index = X_test_con_outliers.index, columns = num_variables)

X_train_con_outliers_norm.to_excel("../data/processed/X_train_con_outliers_norm.xlsx", index = False)
X_test_con_outliers_norm.to_excel("../data/processed/X_test_con_outliers_norm.xlsx", index = False)

normalizador_sin_outliers = StandardScaler()
normalizador_sin_outliers.fit(X_train_sin_outliers)

with open("../models/normalizador_sin_outliers.pkl", "wb") as file:
    pickle.dump(normalizador_sin_outliers,file)

X_train_sin_outliers_norm = normalizador_sin_outliers.transform(X_train_sin_outliers)
X_train_sin_outliers_norm = pd.DataFrame(X_train_sin_outliers_norm, index = X_train_sin_outliers.index, columns = num_variables)

X_test_sin_outliers_norm = normalizador_sin_outliers.transform(X_test_sin_outliers)
X_test_sin_outliers_norm = pd.DataFrame(X_test_sin_outliers_norm, index = X_test_sin_outliers.index, columns = num_variables)

X_train_sin_outliers_norm.to_excel("../data/processed/X_train_sin_outliers_norm.xlsx", index = False)
X_test_sin_outliers_norm.to_excel("../data/processed/X_test_sin_outliers_norm.xlsx", index = False)


## Min-max
min_max_con_outliers = MinMaxScaler()
min_max_con_outliers.fit(X_train_con_outliers)

with open("../models/min_max_con_outliers.pkl", "wb") as file:
    pickle.dump(min_max_con_outliers,file)

X_train_con_outliers_scal = min_max_con_outliers.transform(X_train_con_outliers)
X_train_con_outliers_scal = pd.DataFrame(X_train_con_outliers_scal, index = X_train_con_outliers.index, columns = num_variables)

X_test_con_outliers_scal = min_max_con_outliers.transform(X_test_con_outliers)
X_test_con_outliers_scal = pd.DataFrame(X_test_con_outliers_scal, index = X_test_con_outliers.index, columns = num_variables)

X_train_con_outliers_scal.to_excel("../data/processed/X_train_con_outliers_scal.xlsx", index = False)
X_test_con_outliers_scal.to_excel("../data/processed/X_test_con_outliers_scal.xlsx", index = False)

min_max_sin_outliers = MinMaxScaler()
min_max_sin_outliers.fit(X_train_sin_outliers)

with open("../models/min_max_sin_outliers.pkl", "wb") as file:
    pickle.dump(min_max_sin_outliers,file)

X_train_sin_outliers_scal = min_max_sin_outliers.transform(X_train_sin_outliers)
X_train_sin_outliers_scal = pd.DataFrame(X_train_sin_outliers_scal, index = X_train_sin_outliers.index, columns = num_variables)

X_test_sin_outliers_scal = min_max_sin_outliers.transform(X_test_sin_outliers)
X_test_sin_outliers_scal = pd.DataFrame(X_test_sin_outliers_scal, index = X_test_sin_outliers.index, columns = num_variables)

X_train_sin_outliers_scal.to_excel("../data/processed/X_train_sin_outliers_scal.xlsx", index = False)
X_test_sin_outliers_scal.to_excel("../data/processed/X_test_sin_outliers_scal.xlsx", index = False)

# Selección de características
selection_model = SelectKBest(f_classif, k = 5)
selection_model.fit(X_train_con_outliers_scal, y_train)

ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train_con_outliers_scal), columns = X_train_con_outliers_scal.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test_con_outliers_scal), columns = X_test_con_outliers_scal.columns.values[ix])

with open("../models/feature_selection_k_5.json", "w") as f:
    json.dump(X_train_sel.columns.tolist(), f)


X_train_sel["y_n"] = list(y_train)
X_test_sel["y_n"] = list(y_test)

X_train_sel.to_csv("../data/processed/clean_train.csv", index=False)
X_test_sel.to_csv("../data/processed/clean_test.csv", index=False)

# MACHINE LEARNING
BASE_PATH = "../data/processed"
TRAIN_PATHS = [
    "X_train_con_outliers.xlsx",
    "X_train_sin_outliers.xlsx",
    "X_train_con_outliers_norm.xlsx",
    "X_train_sin_outliers_norm.xlsx",
    "X_train_con_outliers_scal.xlsx",
    "X_train_sin_outliers_scal.xlsx"
]
TRAIN_DATASETS = []
for path in TRAIN_PATHS:
    TRAIN_DATASETS.append(
        pd.read_excel(f"{BASE_PATH}/{path}")
    )

TEST_PATHS = [
    "X_test_con_outliers.xlsx",
    "X_test_sin_outliers.xlsx",
    "X_test_con_outliers_norm.xlsx",
    "X_test_sin_outliers_norm.xlsx",
    "X_test_con_outliers_scal.xlsx",
    "X_test_sin_outliers_scal.xlsx"
]
TEST_DATASETS = []
for path in TEST_PATHS:
    TEST_DATASETS.append(
        pd.read_excel(f"{BASE_PATH}/{path}")
    )

y_train = pd.read_excel(f"{BASE_PATH}/y_train.xlsx")
y_test = pd.read_excel(f"{BASE_PATH}/y_test.xlsx")

results = []
for index, dataset in enumerate(TRAIN_DATASETS):
    model = LogisticRegression(random_state = 42)
    model.fit(dataset, y_train)
    y_pred_train = model.predict(dataset)
    y_pred_test = model.predict(TEST_DATASETS[index])

    results.append(
        {
            "train": accuracy_score(y_train, y_pred_train),
            "test": accuracy_score(y_test, y_pred_test)
        }
    )

best_dataset = 4

hyperparams = {
    "penalty": ["l1", "l2"],
    "tol": [0.0001, 0.001, 0.1],
    "fit_intercept": [True, False],
    "solver": ["liblinear"] 
}

model = LogisticRegression(random_state = 42)
grid = GridSearchCV(model, hyperparams, scoring = "accuracy", cv=5)
grid.fit(TRAIN_DATASETS[best_dataset], y_train)

final_model = grid.best_estimator_
y_pred_train = final_model.predict(TRAIN_DATASETS[best_dataset])
y_pred_test = final_model.predict(TEST_DATASETS[best_dataset])

results.append({
        "train": accuracy_score(y_train, y_pred_train),
        "test": accuracy_score(y_test, y_pred_test),
        "best_params": grid.best_params_
    })

with open("../models/logreg_best_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

with open("../models/final_results.json", "w") as f:
    json.dump(results, f, indent=4)