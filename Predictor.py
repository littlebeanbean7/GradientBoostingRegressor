
import pickle
import pandas as pd
import numpy as np


class Predictor:

    def __init__(self, model_file = 'saved_model/model.sav', sc_X_file = 'saved_model/sc_X.sav', sc_y_file = 'saved_model/sc_y.sav'):
        self.model = pickle.load(open(model_file, 'rb'))
        self.sc_X = pickle.load(open(sc_X_file, 'rb'))
        self.sc_y = pickle.load(open(sc_y_file, 'rb'))

    def predictor(self, patient_id, features):

        # dataprepocessing
        # ## give column names to input features
        name_lst = []
        for i in range(1, 66):
            name = "feature_" + str(i)
            name_lst.append(name)
        features.columns = name_lst

        # ## scaling input features
        features = self.sc_X.transform(np.asarray(features).reshape(1, -1))

        # ## generate one hot key columns.
        lst = []
        for i in range(0, 14):
            if i == patient_id - 2: # Note that we -2 here becuase when training the model, we deleted the 1st dummy variable to avoid dummy variable trap.
                val = 1
            else:
                val = 0
            lst.append(val)

        onehotkey = np.asarray(lst).reshape(1, -1)

        # ## concatenate onehotkey and features into the test array
        test = np.concatenate((onehotkey, features), axis=1)

        # predict

        # ## make prediction
        prediction = self.model.predict(test)

        # ## inverse scaling of the prediction
        prediction = self.sc_y.inverse_transform(prediction)

        # ## take the value out from array
        prediction = prediction[0]

        return prediction




def main():

    # read in a X_test data
    test_df = pd.read_csv("saved_model/X_test.csv", header=None)
    index = test_index
    test = test_df.loc[[index]]

    # get the patient_id from the X_text
    patient_id = int(test.iloc[0, -1])

    # get the features from the X_text
    features = test.iloc[:, :-1]

    # read in a y_test data
    target_df = pd.read_csv("saved_model/y_test.csv", header=None)
    index = test_index

    # get the real target from y_test, just to compare with our prediction
    target = target_df.loc[[index]]
    target = target.iloc[0, 0]

    # call Predictor() to predict
    pred = Predictor()
    prediction = pred.predictor(patient_id, features)

    # print out result
    print('\n')
    print("Here is the records of test sample " + str(test_index) + ":")
    print('Test Index: {} \nPatient ID: {} \nFeatures: \n{} \nTarget: {}'.
          format(test_index, patient_id, features, target))
    print('\n')
    print("Here is the prediction of test sample " + str(test_index) + ":")
    print('Prediction: {}'.format(prediction))

    return prediction

if __name__ == '__main__':

    while True:
        greetings = "We have 14967 test samples, please choose a random integer from 0 to 14966."
        try:
            userinput = int(input(greetings))
        except ValueError:
            print("Sorry, it is not an integer.")
            continue

        test_index = int(userinput)

        if test_index not in range(14967):
            print("Sorry, it is out of range.")
            continue

        else:
            main()
            break