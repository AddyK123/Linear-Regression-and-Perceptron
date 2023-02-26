
import numpy as np
import pandas as pd
import sys

def  get_cost(b0, b1, b2, X):
    cost = 0
    m = len(X.index)
    for row, xi in X.iterrows():
        y_pred = b0 + b1 * xi.age_std + b2 * xi.weight_std
        cost += np.square(y_pred - xi.height)

    cost = cost * (1/2*m) 
    return cost

def gradient(df, B_0, B_age, B_weight, a, n):
    bias_dist = 0
    b1_dist = 0
    b2_dist = 0

    for row, xi in df.iterrows():
        y = df.iloc[row, 5]
        y_pred = B_0 + B_age * xi.age_std + B_weight * xi.weight_std
        bias_dist += y_pred - y
        b1_dist += (y_pred - y)*xi.age_std
        b2_dist += (y_pred - y)*xi.weight_std
    B_0 = B_0 - a*(1/n)*bias_dist
    B_age = B_age - a*(1/n)*b1_dist
    B_weight = B_weight - a*(1/n)*b2_dist

    return B_0, B_age, B_weight

def main():
    data = sys.argv[1]
    output = sys.argv[2]

    #open up the writing file
    f = open(output, "w")
    
    #open up data file, standardize the age and weight, add a zeros column for intercept
    df = pd.read_csv(data, header = None)
    df.columns = ['age', 'weight', 'height']
    df['bias'] = 1
    df['age_std'] = (df['age'] - df['age'].mean()) / df['age'].std()
    df['weight_std'] = (df['weight'] - df['weight'].mean()) / df['weight'].std()
    df = df[['bias', 'age', 'weight', 'age_std', 'weight_std', 'height']]


    #alpha should be an array of the ten different learning rates, do this at the end
    alpha = [.001, .005, .01, .05, .1, .5, 1, 5, 10]
    n = 100

    for a in alpha:
        for __ in range(n):
            B_0 = 0
            B_age = 0
            B_weight = 0
            B_0, B_age, B_weight = gradient(df, B_0, B_age, B_weight, a, n)
        #cost = get_cost(B_0, B_age, B_weight, df)
        #print(cost)
        line = str(a) + ", " + str(n) + ", " + str(B_0) + ", " + str(B_age) + ", " + str(B_weight) +"\n"
        f.write(line)
    
    a = 2.5
    n = 200
    B_0 = 0
    B_age = 0
    B_weight = 0

    B_0, B_age, B_weight = gradient(df, B_0, B_age, B_weight, a, n)
    line = str(a) + ", " + str(n) + ", " + str(B_0) + ", " + str(B_age) + ", " + str(B_weight) +"\n"
    f.write(line)

    f.close 


if __name__ == "__main__":
    main()