import numpy as np
import torch as t
# import tensorboard
# from tensorboard import program

#FIXED INFO
class Meta:
    seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']

    horizons = [6, 8, 18, 13, 14, 48]

    frequencies = [1, 4, 12, 1, 1, 24]

    lookbacks = [2, 3, 4, 5, 6, 7]

    history_size = {
        'Yearly': 1.5,
        'Quarterly': 1.5,
        'Monthly': 1.5,
        'Weekly': 10,
        'Daily': 10,
        'Hourly': 10
    }

    horizons_map = {
        'Yearly': 6,
        'Quarterly': 8,
        'Monthly': 18,
        'Weekly': 13,
        'Daily': 14,
        'Hourly': 48
    }

    frequency_map = {
        'Yearly': 1,
        'Quarterly': 4,
        'Monthly': 12,
        'Weekly': 1,
        'Daily': 1,
        'Hourly': 24
    }

    # denoms = ['AUD', 'BNB', 'BRL', 'BTC', 'ETH', 'EUR', 'RUB', 'USD', 'TRY', 'USDT', 'USDC', 'BIDR', 'IDR', 'GYEN']
    denoms = ['AUD', 'BNB', 'BRL', 'BTC', 'ETH', 'EUR', 'RUB', 'USD', 'TRY', 'USDT', 'USDC', 'IDR']

    cols = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
    
# #Start and Launch Tensorboard
# def tensorboard_launch():
#     tb = program.TensorBoard()
#     tb.configure(argv=[None, '--logdir', 'logs'])
#     tb.launch()

def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result