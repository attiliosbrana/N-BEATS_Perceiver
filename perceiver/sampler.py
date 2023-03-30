import numpy as np
from numba import jit, njit
from datetime import datetime

@njit
def normalize(asts):
    return (asts.T / asts[:, -1]).T

@njit
def weighted_port(asts, weights):
    return (asts.T * weights).sum(axis = 1)

@njit
def clean_up_weights(weights, threshold = 0.025):
    weights[weights < 0.025] = 0
    return weights / weights.sum()

@njit
def random_weights(num_of_tokens):
    return np.random.dirichlet(np.ones(num_of_tokens))

@njit
def train_window(dates_array, eligibles, train_cutoff, start, end, insample_size, outsample_size):
    main_token_idx = choice(eligibles)
    token_start, token_finish = start[main_token_idx], end[main_token_idx]
    min_start, max_end = token_start + 1, min(train_cutoff, token_finish) + 1
    cut_point = np.random.randint(min_start, max_end)
    backcast_start, backcast_end = max(token_start, cut_point - insample_size), cut_point
    forecast_start, forecast_end = cut_point, min(max_end, cut_point + outsample_size)
    date = dates_array[cut_point - 1]
    return main_token_idx, cut_point, backcast_start, backcast_end, forecast_start, forecast_end, date

@njit
def test_window(dates_array, eligibles, train_cutoff, start, end, insample_size, outsample_size):
    main_token_idx = choice(eligibles)
    token_start, token_finish = start[main_token_idx], end[main_token_idx]
    min_start, max_end = max(token_start, train_cutoff) + 1, token_finish + 1
    cut_point = np.random.randint(min_start, max_end)
    backcast_start, backcast_end = max(token_start, cut_point - insample_size), cut_point
    forecast_start, forecast_end = cut_point, min(max_end, cut_point + outsample_size)
    date = dates_array[cut_point - 1]
    return main_token_idx, cut_point, backcast_start, backcast_end, forecast_start, forecast_end, date

@njit
def tokens(shape):
    return np.random.randint(1, max(min([41, shape]), 2))

@njit
def idxs(compatibles, num_of_tokens):
    return np.random.choice(compatibles, num_of_tokens, replace = False)

@njit
def idxs(main_token_idx, compatibles, num_of_tokens):
    tokens = np.random.choice(compatibles, num_of_tokens, replace = False)
    return np.append(tokens, main_token_idx)

@njit
def choice(x):
    return np.random.choice(x)

@njit
def slices(data, i, start, end):
    return data[i, start:end]

@njit
def append(a, b):
    return np.append(a, b)

# @njit
# def build_random_portfolios(main_token_idx, compatibility, series, backcast_start, backcast_end,
#                      forecast_start, forecast_end, date):
    
#     compatibles = compatibility[main_token_idx]
#     num_of_tokens, tokens_idx = random_assets(compatibles, main_token_idx)
    
#     backcasts = slices(series, tokens_idx, backcast_start, backcast_end)
#     forecasts = slices(series, tokens_idx, forecast_start, forecast_end)

#     normalized_backcasts = normalize(backcasts)
#     normalized_forecasts = normalize(forecasts)

#     weights = clean_up_weights(random_weights(num_of_tokens + 1))

#     portfolio_backcast = weighted_port(normalized_backcasts, weights)
#     portfolio_forecast = weighted_port(normalized_forecasts, weights)
    
#     portfolio_backcast = append(portfolio_backcast, date)
#     return portfolio_backcast, portfolio_forecast

@njit
def build_random_portfolios(main_token_idx, compatibility, series, backcast_start, backcast_end,
                     forecast_start, forecast_end, date):
    
    compatibles = compatibility[main_token_idx]
    num_of_tokens, tokens_idx = random_assets(compatibles, main_token_idx)
    
    backcasts = slices(series, tokens_idx, backcast_start, backcast_end)
    normalized_backcasts = normalize(backcasts)

    forecasts = slices(series, tokens_idx, forecast_start, forecast_end)
    normalized_forecasts = (forecasts.T / backcasts[:, -1]).T

    weights = clean_up_weights(random_weights(num_of_tokens + 1))

    portfolio_backcast = weighted_port(normalized_backcasts, weights)
    portfolio_forecast = weighted_port(normalized_forecasts, weights)
    
    portfolio_backcast = append(portfolio_backcast, date)
    return portfolio_backcast, portfolio_forecast

@njit
def random_assets(compatibles, main_token_idx):
    num_of_tokens = tokens(compatibles.shape[0])
    tokens_idx = idxs(main_token_idx, compatibles, num_of_tokens)
    return num_of_tokens, tokens_idx

@njit
def get_train_batch(dates, dates_array, train_cutoff, series, start_offsets, finish_offsets, eligibles,
                    compatibility, insample_size, outsample_size, batch_size):
    
    #Initiate batch
    date_array_len = 3
    insample = np.zeros((batch_size, insample_size + date_array_len))
    insample_mask = np.zeros((batch_size, insample_size + date_array_len))
    outsample = np.zeros((batch_size, outsample_size))
    outsample_mask = np.zeros((batch_size, outsample_size))
    
    for i in range(batch_size):

        #Get a window
        main_token_idx, cut_point, backcast_start, backcast_end, forecast_start, forecast_end, date = \
        train_window(dates_array, eligibles, train_cutoff, start_offsets, finish_offsets,
                     insample_size, outsample_size)
        
        #Build random portfolios
        portfolio_backcast, portfolio_forecast = build_random_portfolios(main_token_idx, compatibility,
                                                                         series, backcast_start, backcast_end,
                                                                         forecast_start, forecast_end, date)
        #Modify the batch array
        insample[i, -len(portfolio_backcast):] = portfolio_backcast
        insample_mask[i, -len(portfolio_backcast):] = 1.0
        outsample[i, :len(portfolio_forecast)] = portfolio_forecast
        outsample_mask[i, :len(portfolio_forecast)] = 1.0
    
    return insample, insample_mask, outsample, outsample_mask

@njit
def get_test_batch(dates, dates_array, train_cutoff, series, start_offsets, finish_offsets, eligibles,
                    compatibility, insample_size, outsample_size, batch_size):
    
    #Initiate batch
    date_array_len = 3
    insample = np.zeros((batch_size, insample_size + date_array_len))
    insample_mask = np.zeros((batch_size, insample_size + date_array_len))
    outsample = np.zeros((batch_size, outsample_size))
    outsample_mask = np.zeros((batch_size, outsample_size))
    
    for i in range(batch_size):

        #Get a window
        main_token_idx, cut_point, backcast_start, backcast_end, forecast_start, forecast_end, date = \
        test_window(dates_array, eligibles, train_cutoff, start_offsets, finish_offsets,
                     insample_size, outsample_size)
        
        #Build random portfolios
        portfolio_backcast, portfolio_forecast = build_random_portfolios(main_token_idx, compatibility,
                                                                         series, backcast_start, backcast_end,
                                                                         forecast_start, forecast_end, date)
        #Modify the batch array
        insample[i, -len(portfolio_backcast):] = portfolio_backcast
        insample_mask[i, -len(portfolio_backcast):] = 1.0
        outsample[i, :len(portfolio_forecast)] = portfolio_forecast
        outsample_mask[i, :len(portfolio_forecast)] = 1.0
    
    return insample, insample_mask, outsample, outsample_mask