# BitCoin_Predictor

Directories:
cryptocurrency-info:
.venv

File Structure:
cryptocurrency-info/BTC.csv
cryptocurrency-info/DOGE.csv
cryptocurrency-info/ETH.csv
cryptocurrency-info/XMR.csv
cryptocurrency-info/new_scraper.py

raw_df.csv


Predictor.py

def split_for_training(datafile, predictionDays,independent_array, dependent_array, test_percent):
    #Description: Read in the data from the given csv file so that it can be used by the program
    #Input: datafile: the DataFrame containing the csv file information
    #       predictionDays, the total number of days that you are wanting to predict
    #       independent_array, the 'x' value array that will be used to predict the price of bitcoin
    #       dependent_array, the 'y' value array that holds the corresponding prices from the predictor
    #       test_percent: the value that is a percentage (between 0 and 1) that represents the portion of 'x' that will be reserved for testing
    #Returns: returns a DataFrame type named datafile so that we can compute information from it

def create_svm(xtrain, xtest, ytrain, ytest):
    #Description: Builds a Support Vector Regression Model for predicting the price of bitcoin
    #Input: xtrain: array containing all of the 'x' values that are for training
    #       xtest: array containing all of the 'x' values that are for testing
    #       ytrain: array containing all of the 'y' values corresponding to 'x' for training
    #       ytest: array containing all of the 'y' values corresponding to 'x' for testing
    #Returns: returns a Support Vector Regression Model 

def predict_for_n_entries(datafile, predictionEntries):
    #Description: Builds and predicts the bitcoin prices for a given number 'n' days
    #Input: datafile: The file that contains all of the values from the csv file
    #       predictionEntries: The number of entries that you want to predict for
    #Returns: returns a Support Vector Regression Model 

def read_in_data(filename):
#Description: Read in the data from the given csv file so that it can be used by the program
#Input: filename: is the filepath from the environment to the datafile. 
#Returns: returns a DataFrame type named datafile so that we can compute information from it
