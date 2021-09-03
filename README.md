# sklearn_workflow
Notes I've gathered to help reduce time reading documentation repeatedly when using sklearn
# General workflow with sklearn

## 1. Get data
Scrape from or download a webpage or load from a database 

### scrape
```python
import beautifulsoup4 as bs
import requests as r

url = 'https://example.com'
res = r.get(url)
if (res != 200)
    print(f"{res}: Did not get proper response from {url}")
    exit()

# WIP
# ...
```

### from a database
```python
# WIP
# SQL libraries and boilerplate here
```

### create artificial data
```python
# WIP
# numpy, seaborn and sklearn can generate random data easily
```

## 2. Read data
Read data into a pandas dataframe if already in a proper format, otherwise format data (and e.g. write into a .csv)
```python
import numpy as np
import pandas as pd
```

### from a .csv
```python
csv_path = 'resources/data.csv'
df = pd.read_csv(csv_path)
```

### from a list, array or dictionary
```python
# list of lists example
data = [[2, 4, 6, 8, 10],
        [-0.5, 1.1, -2.98, 0, 3.05],
        "john, jane, alice, bob, charlie".split()
        ]
df = pd.DataFrame(data, index = data[2])

# dict example
data = {'column1': np.arange(1,65),
        'column2': np.arange(50,101),
        'column3': np.linspace(0.0, 1.0, 7)
        }
df = pd.DataFrame(data)
```

### more examples...

## 3. Explore data
Get an overview of the data on hand by visualizing it

Quick methods off a `pandas.DataFrame` to use in Jupyter Notebook:
`.head()`
`.info()`
`.describe()`

Quick visualization methods
```python
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline  # if on Jupyter Notebook

sns.pairplot(df)    # takes a while if dataset is huge
sns.regplot(df[['column1', 'column2']])
# WIP
# more useful plots...
```


## 4. Clean data
If necessary
```python
df.dropna(axis=0)   # removes rows that have NaN
df.dropna(axis=1)   # removes columns that have NaN

# fill each variable's NaN with its mean for the whole dataset
for col in df.columns:
    df[col].fillna(value = df[col].mean())
```

## 5. [Scale data](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-data)
Some models assume normally distributed data, i.e. zero mean and unit variance

Used by e.g. LogisticRegression or SVC
```python
from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(df)
scaled_data = scaler.transform(df)
```

Or to do the above easier, use a pipeline
```python
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(preprocessing.StandardScaler(), LogisticRegression())
pipe.fit(df)
```

## 6. Build model
Once you have formatted the data to build a model out of...

Split data
```python
from sklearn.cross_validation import train_test_split

X = df.drop('labels', axis=1)
y = df['labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

Select model
- predicts a datapoint's target **value** based on coefficients constructed out of its other features:
`sklearn.linear_model.LinearRegression`

- predicts a datapoint's **binary label** based on its other features:
`sklearn.linear_model.LogisticRegression`
or (  `sklearn.model_selection.GridSearchCV` recommended: )
`sklearn.svm.SVC`

- predicts a datapoint's **target class** based on the "nearby" datapoints' target class:
`sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)`

- predicts a **target class** based on splits made in a decicion tree:
`sklearn.tree.DecisionTreeClassifier`

- predicts a **target class** based on *random* splits made in `x` `DecisionTreeClassifier`s:
`sklearn.ensemble.RandomForestClassifier(n_estimators=x)`


Fit data and get predictions
```python
model_instance.fit(X_test)
predictions = model_instance.predict()
```


## 7. Inspect error metrics
```python
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))
print(classificiation_report(y_test, predictions))
```
