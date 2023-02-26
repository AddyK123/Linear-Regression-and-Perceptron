import pandas as pd
import numpy as np
import sys

def main():
    data = sys.argv[1]
    output = sys.argv[2]

    #open up the writing file
    f = open(output, "w")
    
    #open up data file and make add x0 column
    df = pd.read_csv(data, header = None)
    df.columns = ['feature_1', 'feature_2', 'label']
    df['bias'] = 1
    df = df[['bias', 'feature_1', 'feature_2', 'label']]
    
    #weights set up
    w0 = 0
    w1 = 0
    w2 = 0
    
    line = str(w0) + ", " + str(w1) + ", " + str(w2) +"\n"
    f.write(line)

    conv = False
    while not conv:
        conv = True
        for row, xi in df.iterrows():
            y = df.iloc[row, 3]
            y_pred = xi.label * np.sign((xi.bias* w0) + (xi.feature_1* w1) + (xi.feature_2* w2))
            if y_pred <= 0:
                conv = False
                w0 += y*xi.bias
                w1 += y*xi.feature_1
                w2 += y*xi.feature_2
        line = str(w0) + ", " + str(w1) + ", " + str(w2) +"\n"
        f.write(line)



    f.close()

    '''YOUR CODE GOES HERE'''


if __name__ == "__main__":
    """DO NOT MODIFY"""
    main()