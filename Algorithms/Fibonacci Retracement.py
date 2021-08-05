#Fibonacci Retracement
from PythonDataProcessing import DataRetrieval as DR
from numpy import NaN


data = DR.get_stock_data('DIS', period='1y')









def add_fibonacci_retracement(data):
    '''
    This function takes a data frame of historic data and finds the fibonacci values for an interval of time and adds the values to the data frame.

    Input:
        *data* (required) - The dataframe of historic data the fibonacci levels will be found from.
    
    Output:
        *data* - A data frame with the fibonacci levels added
    
    Examples:
        add_fibonacci_retracement(data_source)
    '''

    #Currently an interval of 50
    fib_interval = 50
    i = 0

    fib_interval = fib_interval -1

    FRL = [NaN] *fib_interval
    FR236 = [NaN] *fib_interval
    FR382 = [NaN] *fib_interval
    FR5 = [NaN] *fib_interval
    FR618 = [NaN] *fib_interval
    FR786 = [NaN] *fib_interval
    FRH = [NaN] *fib_interval

    fib_interval = fib_interval +1

    while i+fib_interval <= data.shape[0]:
        #Breaks up data to look at only a specified interval before a given data point
        fib_data = data[:][i : i+fib_interval]

        #Calculates fibonacci values and adds them to lists
        high = fib_data['High'][:].max()
        low = fib_data['Low'][:].min()

        difference = high - low

        FRL.append(low)
        FR236.append(low + (difference * .236))
        FR382.append(low + (difference * .382))
        FR5.append(low + (difference * .5))
        FR618.append(low + (difference * .618))
        FR786.append(low + (difference * .786))
        FRH.append(high)

        i = i+1
    
    #Adds fibonacci levels to data frame
    data['FRL'] = FRL
    data['FR236'] = FR236
    data['FR382'] = FR382
    data['FR5'] = FR5
    data['FR618'] = FR618
    data['FR786'] = FR786
    data['FRH'] = FRH











add_fibonacci_retracement(data)
print(data)