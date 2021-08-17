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
    
    fib_interval = fib_interval -1

    FRL = [NaN] *fib_interval
    FR236 = [NaN] *fib_interval
    FR382 = [NaN] *fib_interval
    FR5 = [NaN] *fib_interval
    FR618 = [NaN] *fib_interval
    FR786 = [NaN] *fib_interval
    FRH = [NaN] *fib_interval

    fib_interval = fib_interval +1

    #Finds the lowest and highest value over the given interval for each row in the data frame
    lows = [min(data['Low'][j-fib_interval:j]) for j in range(fib_interval, len(data) +1)]
    highs = [max(data['High'][j-fib_interval:j]) for j in range(fib_interval, len(data) +1)]

    for i in range(0, len(lows)):
        difference = highs[i] - lows[i]

        FRL.append(lows[i])
        FR236.append(lows[i] + (difference * .236))
        FR382.append(lows[i] + (difference * .382))
        FR5.append(lows[i] + (difference * .5))
        FR618.append(lows[i] + (difference * .618))
        FR786.append(lows[i] + (difference * .786))
        FRH.append(highs[i])
    
    #Adds fibonacci levels to data frame
    data['FRL'] = FRL
    data['FR236'] = FR236
    data['FR382'] = FR382
    data['FR5'] = FR5
    data['FR618'] = FR618
    data['FR786'] = FR786
    data['FRH'] = FRH


def test_add_FR():
    # Parameters
    ticker = 'AAPL'
    data_path = ticker+'_FR.csv'
    interval = 50

    # Loading test data
    dummy_data = pandas.read_csv(data_path, index_col = 'Date')

    # Getting Fibonacci Retracement Values
    add_FR(dummy_data)
    frL = dummy_data['FRL'].values
    fr236 = dummy_data['FR236'].values
    fr382 = dummy_data['FR382'].values
    fr5 = dummy_data['FR5'].values
    fr618 = dummy_data['FR618'].values
    fr786 = dummy_data['FR786'].values
    frH = dummy_data['FRH'].values

    # Checking with dummy CSV data
    assert all(np.round(dummy_data['L'][interval-1:], 4) == np.round(frL[interval-1:], 4))
    assert all(np.round(dummy_data['236'][interval-1:], 4) == np.round(fr236[interval-1:], 4))
    assert all(np.round(dummy_data['382'][interval-1:], 4) == np.round(fr382[interval-1:], 4))
    assert all(np.round(dummy_data['5'][interval-1:], 4) == np.round(fr5[interval-1:], 4))
    assert all(np.round(dummy_data['618'][interval-1:], 4) == np.round(fr618[interval-1:], 4))
    assert all(np.round(dummy_data['786'][interval-1:], 4) == np.round(fr786[interval-1:], 4))
    assert all(np.round(dummy_data['H'][interval-1:], 4) == np.round(frH[interval-1:], 4))









add_fibonacci_retracement(data)
print(data)
