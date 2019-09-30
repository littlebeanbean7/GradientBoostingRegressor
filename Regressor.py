import glob
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Please change the path to your own working directory
file_list = list(glob.glob('data/' + "*.csv*"))

class Regressor:
    def __init__(self, file_list=file_list):
        file_list.sort()
        self.df_list = []
        for f in file_list:
            df = pd.read_csv(f).drop('Unnamed: 0', 1) # remove first column
            df['patient_id'] = f.split('/')[-1].split('_')[0] # get patient_id
            self.df_list.append(df)

    def encodor(self, data):
        # one hot encode the patient id
        labelencoder = LabelEncoder()
        data[:,-1] = labelencoder.fit_transform(data[:,-1])
        onehotencoder = OneHotEncoder(categorical_features = [-1])
        data = onehotencoder.fit_transform(data).toarray()
        return data

    def dataprepocessing(self):
        # concat all tables to one df
        df = pd.concat(self.df_list)
        # remove rows with inf, -inf
        df = df.replace([np.inf, -np.inf], np.nan)
        # remove rows  with missing value
        df = df.dropna()
        # remove rows with 0 values
        df = df[(df != 0).all(1)]
        df['patient_id'] = df['patient_id'].astype('int64')
        df = df.sort_values(by=['patient_id'])
        df = df.reset_index(drop=True) # 67 columns

        # separate features and targets, change df to array
        X = df.iloc[:,1:].values # 65 features + 1 patient id = 66 columns
        y = df.iloc[:,0].values # 1 column

        # split training and test set # Note: only use 1/10 of the data, due to limited computational resources
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.03, train_size = 0.07, random_state=0)

        # save test data (used for testing our scripts)
        np.savetxt("saved_model/X_test.csv", X_test, delimiter=",")
        np.savetxt("saved_model/y_test.csv", y_test, delimiter=",")

        # scale training set's 65 features and targets
        sc_X = StandardScaler()
        X_train[:, np.s_[0:65]] = sc_X.fit_transform(X_train[:, np.s_[0:65]])
        sc_y = StandardScaler()
        y_train = sc_y.fit_transform(y_train.reshape(-1,1))

        # save scaler
        pickle.dump(sc_X, open('saved_model/sc_X.sav', 'wb')) #scaler of the 65 features
        pickle.dump(sc_y, open('saved_model/sc_y.sav', 'wb'))

        # one hot key encoding for X_train
        X_train = self.encodor(X_train)

        # remove the first column to avoid dummy variable trap
        X_train = X_train[:, 1:]

        # save train data, just to take a look
        np.savetxt("saved_model/X_train.csv", X_train, delimiter=",")
        np.savetxt("saved_model/y_train.csv", y_train, delimiter=",")

        return X_train, y_train

    def fit(self):
        X_train, y_train = self.dataprepocessing()

        #train the model with whole training set
        params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)

        # evaluate the model with 5-fold CV
        scores = cross_val_score(model, X_train, y_train, cv=5)
        df_scores = pd.DataFrame(scores)
        df_scores.index.name = "CV round"
        df_scores = df_scores.T
        df_scores["mean"] = df_scores.mean(axis=1)
        df_scores["std"] = df_scores.std(axis=1)

        # save the model
        pickle.dump(model, open('saved_model/model.sav', 'wb'))

        # save the evaluation
        with open("saved_model/cv_score.txt", 'w') as handle:
            handle.write(str(df_scores))
            handle.write("\n")
        return df_scores

## call 'Regressor'
def main():
    model = Regressor()
    model.fit()

if __name__ == '__main__':
    main()


