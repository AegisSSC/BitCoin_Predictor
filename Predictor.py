#https://thecleverprogrammer.com/2020/05/23/bitcoin-price-prediction-with-machine-learning/
#TO DO: 
# Build a Visualizer 
# Build a Scraper to read in data
# Properly Segment and Generalize the Code
# Add SQL/Database Support


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

#Description: Read in the data from the given csv file so that it can be used by the program
#Input: filename: is the filepath from the environment to the datafile. 
#Returns: returns a DataFrame type named datafile so that we can compute information from it
def read_in_data(filename):
    datafile = pd.read_csv(filename)

    #print the first and last 5 rows
    datafile.head()
    datafile.tail()

    #Remove the first column that starts with Date so that we only have price information. (the model in this instance is date agnostic)
    datafile.drop(['Date'], 1, inplace = True)

    #print the updated head and tail now that we have removed the date column
    datafile.head()
    datafile.tail()

    return datafile

#Description: Read in the data from the given csv file so that it can be used by the program
#Input: datafile: the DataFrame containing the csv file information
#       predictionDays, the total number of days that you are wanting to predict
#       independent_array, the 'x' value array that will be used to predict the price of bitcoin
#       dependent_array, the 'y' value array that holds the corresponding prices from the predictor
#       test_percent: the value that is a percentage (between 0 and 1) that represents the portion of 'x' that will be reserved for testing
#Returns: returns a DataFrame type named datafile so that we can compute information from it
def split_for_training(datafile, predictionDays,independent_array, dependent_array, test_percent):
    xtrain, xtest, ytrain, ytest = train_test_split(independent_array, dependent_array, test_size = test_percent)
    predictionDays_array = np.array(datafile.drop(['Prediction'],1))[-predictionDays:]
    print(predictionDays_array)
    return xtrain, xtest, ytrain, ytest, predictionDays_array

#Description: Builds a Support Vector Regression Model for predicting the price of bitcoin
#Input: xtrain: array containing all of the 'x' values that are for training
#       xtest: array containing all of the 'x' values that are for testing
#       ytrain: array containing all of the 'y' values corresponding to 'x' for training
#       ytest: array containing all of the 'y' values corresponding to 'x' for testing
#Returns: returns a Support Vector Regression Model 
def create_svm(xtrain, xtest, ytrain, ytest):
    #create the Support Vector Regression
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.00001)
    svr_rbf.fit(xtrain, ytrain)

    #now that the model is trained, test the model
    svr_rbf_confidence = svr_rbf.score(xtest,ytest)
    print('SVR_RBF accuracy (closer to 1 the better) = ',svr_rbf_confidence)
    return svr_rbf

#Description: Builds and predicts the bitcoin prices for a given number 'n' days
#Input: datafile: The file that contains all of the values from the csv file
#       predictionDays: The number of days that you want to predict for
#Returns: returns a Support Vector Regression Model 
def predict_for_n_days(datafile, predictionDays):
    datafile['Prediction'] = datafile[['Price']].shift(-predictionDays)
    #show the first and last 5 Rows
    datafile.head()
    datafile.tail()
    plt.plot(range(0,367), datafile['Price'])
    #create the independent data set
    x = np.array(datafile.drop(['Prediction'],1))
    #remove the last 'n' rows: where 'n' = predictionDays
    x = x[:len(datafile)-predictionDays]
    # print("The independent data set is" + x)
    
    #create the depedent data set
    y = np.array(datafile['Prediction'])
    #Get all values except for the last 'n' rows
    y = y[:-predictionDays]
    # print("The dependent data set is" + y)

    #split the data so that 20% is used for testing
    xtrain, xtest, ytrain, ytest, predictionDays_array = split_for_training(datafile, predictionDays, x, y, test_percent=0.2)

    #create a support vector machine
    svr_rbf = create_svm(xtrain, xtest, ytrain, ytest)
    svm_prediction = svr_rbf.predict(xtest)
    # print(svm_prediction)
    # print()
    # print(ytest)

    #Plot the Model Predictions for the next 'n' days against the total graph
    
    svm_prediction = svr_rbf.predict(predictionDays_array)
    # plt.plot(xtest,svm_prediction, label = "Predicted Outcome")
    plt.plot(range(367-predictionDays,367), svm_prediction, label = "prediction")
    plt.plot(range(367-predictionDays, 367), datafile.tail(predictionDays))
    plt.xlabel('x - time interval (days) ')
    plt.ylabel('y - value of the stock ($USD)')
    plt.title("Predicting the expected value of a cryptocurrency")
    plt.legend()
    plt.show()


    # print(svm_prediction)
    # print()
    # #Print the actual price for 'n' Days
    # print(datafile.tail(predictionDays))


def main():
    #Get the filename that they would like to input
    filename = 'bitcoin-info/bitcoin.csv'
    # filename = input("Please enter the filepath you would like to gather information from: ")
    datafile = read_in_data(filename)
    #Get the number of days you would like to predict for
    predictionDays = 30
    # predictionDays = int(input("Please enter the the number of days you would like to predict: "))
    predict_for_n_days(datafile, predictionDays)

    
if __name__ == "__main__":
    main()  