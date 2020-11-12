#load in data as before

import pandas as pd
titanic_train_path = '/Users/mrharwood/Desktop/Titanic/train.csv'
titanic_data = pd.read_csv(titanic_train_path)

#python3 -m pip install category_encoders
#python3 -m pip install xgboost==0.7.post4
#categorical encodings
#split first to avoid leakage
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

y = titanic_data.Survived
X_full = titanic_data


#drop cabins because its mostly empty

X_dropped = X_full.drop(['Cabin'], axis = 1)

def get_data_splits(dataframe, valid_fraction=0.1):

    #dataframe = dataframe.sort_values('click_time')
    valid_rows = int(len(dataframe) * valid_fraction)
    train_X = dataframe[:-valid_rows]
    # valid size == test size, last two sections of the data
    valid_X = dataframe[-valid_rows:]
    test = dataframe[-valid_rows:]
    
    return train_X, valid_X, test

train_X, valid_X, test = get_data_splits(X_dropped)

numerical_cols = [cname for cname in train_X.columns if 
                 train_X[cname].dtype in ['int64', 'float64']]

categorical_cols = [cname for cname in train_X.columns if
                    train_X[cname].dtype == "object"]

numerical_cols.remove("Survived")

#adding inteactions between categorical cols
import itertools 

interactions = pd.DataFrame(index=train_X.index)

for col1, col2 in itertools.combinations(categorical_cols, 2):
    new_col_name = str(col1 + '_' + col2)
    new_values = train_X[col1].map(str) + "_" + train_X[col2].map(str)
    interactions[new_col_name] = new_values

train_int = train_X.join(interactions)
valid_int = valid_X.join(interactions)

new_cats = [] 
for cname in train_int.columns:
    if train_int[cname].dtype == "object":
        new_cats.append(cname)

#imputing train and the matching valid
num_impute = SimpleImputer(strategy='mean')
cat_impute = SimpleImputer(strategy='constant')

cat_impute.fit(train_int[new_cats])
num_impute.fit(train_int[numerical_cols])

train_cat_imp_cols = pd.DataFrame(cat_impute.transform(train_int[new_cats]))
train_num_imp_cols = pd.DataFrame(num_impute.transform(train_int[numerical_cols]))

valid_cat_imp_cols = pd.DataFrame(cat_impute.transform(valid_int[new_cats]))
valid_num_imp_cols = pd.DataFrame(num_impute.transform(valid_int[numerical_cols]))

#adding names back
train_cat_imp_cols.columns = new_cats
train_num_imp_cols.columns = numerical_cols

valid_cat_imp_cols.columns = new_cats
valid_num_imp_cols.columns = numerical_cols

#joining
train_imp = train_int.join(train_cat_imp_cols.add_suffix("_imp"))
train_imp = train_imp.join(train_num_imp_cols.add_suffix("_imp"))

valid_imp = valid_int.join(valid_cat_imp_cols.add_suffix("_imp"))
valid_imp = valid_imp.join(valid_num_imp_cols.add_suffix("_imp"))

#removing non imputed
non_imp = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Embarked', 'Name_Sex', 'Name_Ticket','Name_Embarked', 'Sex_Ticket', 'Sex_Embarked', 'Ticket_Embarked']
train_imp = train_imp.drop(non_imp, axis = 1)
valid_imp = valid_imp.drop(non_imp, axis = 1)

#now cat_encoding

imp_cats = [cname for cname in train_imp.columns if train_imp[cname].dtype == "object"]

cat_encoder = ce.CatBoostEncoder()
cat_encoder = cat_encoder.fit(train_imp[imp_cats], train_imp['Survived'])

train_enc = train_imp.join(cat_encoder.transform(train_imp[imp_cats]).add_suffix('_enc'))
valid_enc = valid_imp.join(cat_encoder.transform(valid_imp[imp_cats]).add_suffix('_enc'))

train_enc = train_enc.drop(imp_cats, axis = 1)
valid_enc = valid_enc.drop(imp_cats, axis = 1)

#ok now lets try some pca to choose the best 5 feaatures

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

X, y = train_enc[train_enc.columns.drop("Survived")], train_enc['Survived']

logistic = LogisticRegression(C=1, penalty="l1", solver='liblinear', random_state=0)
logistic.fit(X, y)

model = SelectFromModel(logistic, prefit=True)

X_new = model.transform(X)

selected_features = pd.DataFrame(model.inverse_transform(X_new), 
                                 index=X.index,
                                 columns=X.columns)

selected_columns = selected_features.columns[selected_features.var() != 0]

best_features_train = train_enc[selected_columns]
best_features_valid = valid_enc[selected_columns]

#lets try fitting a model to these selected features

from xgboost import XGBRegressor
from sklearn.metrics import roc_auc_score

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(best_features_train, train_enc['Survived'], 
             early_stopping_rounds=5, 
             eval_set=[(best_features_valid, valid_enc['Survived'])], 
             verbose=False)

preds = my_model.predict(best_features_valid)

print(roc_auc_score(valid_enc['Survived'], preds))

#lets get this working on the test data

titanic_test_path = '/Users/mrharwood/Desktop/Titanic/test.csv'
titanic_test = pd.read_csv(titanic_test_path)
test = titanic_test.copy()

#transform to match

test = test.drop(['Cabin'], axis = 1)

test_interactions = pd.DataFrame(index=test.index)

for col1, col2 in itertools.combinations(categorical_cols, 2):
    new_col_name = str(col1 + '_' + col2)
    new_values = test[col1].map(str) + "_" + test[col2].map(str)
    test_interactions[new_col_name] = new_values

test_int = test.join(test_interactions)

test_cat_imp_cols = pd.DataFrame(cat_impute.transform(test_int[new_cats]))
test_num_imp_cols = pd.DataFrame(num_impute.transform(test_int[numerical_cols]))

test_cat_imp_cols.columns = new_cats
test_num_imp_cols.columns = numerical_cols

test_imp = test_int.join(test_cat_imp_cols.add_suffix("_imp"))
test_imp = test_imp.join(test_num_imp_cols.add_suffix("_imp"))

test_imp = test_imp.drop(non_imp, axis = 1)

#matching encoding
test_enc = test_imp.join(cat_encoder.transform(test_imp[imp_cats]).add_suffix('_enc'))

test_enc = test_enc.drop(imp_cats, axis = 1)

best_features_test = test_enc[selected_columns]

#predicticting and saving to csv

my_preds = [int(round(num, 0)) for num in my_model.predict(best_features_test)]

print(selected_columns)


#output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': my_preds})
#output.to_csv('/Users/mrharwood/Desktop/Titanic/submission.csv', index=False)

#79% and 3000 out of 17000!









