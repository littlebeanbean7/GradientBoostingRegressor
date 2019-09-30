# GradientBoostingRegressor

1. In your working directory:
1.1 Put the data/ folder into this directory
1.2 Create a directory saved_model/
1.3 Create a directory code/, and put the 2 scripts into the directory: Regressor.py and Predictor.py
1.4 To generate the model, run $python3 Regressor.py
1.5 To predict, run $python3 Predictor.py

2. About Regressor.py 
This script produces the following 8 files to the saved_model/ directory:

1) model.sav: The GradientBoostingRegressor fitted by X_train.
2) sc_X.sav: The StandardScaler fitted by X_train.
3) sc_y.sav: The StandardScaler fitted by y_train.
4) cv_score.txt: The model performance evaluated by 5 fold cross validation and the final performance (i.e., the averaged result of each cross validation)
5) 6) X_test.csv and y_test csv: Some test samples that are not used in the training and validation process. These samples are dedicated to test whether our program is running fine. 
7) 8) X_train.csv and y_train csv: These are the datasets produced after all the data processing, just output a copy of them to check out.

Please note that during training, I removed rows with missing values, zero values and infinite values, and just used 10% of the remaining data due to limited computational resource.
I used one hot key encoding for the patient id variable, and removed the first dummy variable to avoid dummy variable trap. 

3. About Predictor.py
This script generates prediction based on user’s choice of test data sample. it takes one input a time.

First, it defines a class called Predictor(). It makes predictions based on two parameters: patient_id (int) and record (data frame).

It has 2 methods:
1) the __init__ method loads model.sav, sc_X.sav, sc_y.sav
2) the predictor method makes the prediction.

Second, it has a main() function, with which users can test the Predictor() without needing to manually input a lot of records. 

When running the script, the program would ask the user to input a random number, which would refers to the index of a sample in the X_test.csv. The program would then fetch the records (patient id and features) and generate the prediction.
