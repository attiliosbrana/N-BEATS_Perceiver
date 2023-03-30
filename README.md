# N-BEATS Perceiver

This repository contains the code and resources for the paper "N-BEATS Perceiver: A Novel Approach for Robust Cryptocurrency Portfolio Forecasting"[1]. It relies on the Binance Portfolio Forecasting Hourly VWAP Dataset, which is available in the OSF Repository [2] with accompanying code in its GitHub repository [3].

## Repository Structure

- `train_N-BEATS_Perciever.ipynb`: Includes all the code to train the 54 instances of the N-BEATS Perceiver used in the paper. It also includes the code that automatically downloads the OSF data file for training. The file also includes all the hyperparameters used for training the models of the N-BEATS Perceiver paper.
- `./data`: An empty folder intended to host the training data available in the OSF repository.
- `./perceiver`: Contains all the source code necessary for the Perceiver model, N-BEATS modifications, losses, training schedule, optimizers, utils, portfolio sampling source code, etc.
- `./model_checkpoints`: An empty folder intended to host the trained weights of the N-BEATS Perceiver model as they get trained.

## Paper Abstract

Cryptocurrencies are well-known for their high volatility and unpredictability, posing a challenge for forecasting using traditional methods. To address this issue, we explore variations of the N-BEATS deep learning (DL) architecture by adding convolutional network layers, Transformer mechanisms, and the Mish activation function, and propose a novel approach for forecasting cryptocurrency portfolios. Our comprehensive evaluation demonstrates that our model variations outperform other DL and traditional forecasting methods in numerous evaluation metrics, making them powerful tools for predicting cryptocurrency prices and portfolios in the rapidly-evolving cryptocurrency market. Furthermore, our newly proposed N-BEATS Perceiver model, a Transformer-based N-BEATS variation, exhibits a robust risk profile with less downside compared to other models and performs exceptionally well when evaluated using the TOPSIS method across a wide range of portfolio evaluation parameters. These results underscore the potential of our approach and specifically highlight the N-BEATS Perceiver's potential for selecting portfolios and forecasting cryptocurrency prices, offering valuable insights into the development of more accurate and reliable models for cryptocurrency forecasting.

## References

[1] Sbrana, A., & Lima de Castro, P. A. (2023, February 23). N-BEATS Perceiver: A Novel Approach for Robust Cryptocurrency Portfolio Forecasting (Version 1) [Preprint]. Research Square. https://doi.org/10.21203/rs.3.rs-2618277/v1

[2] Sbrana, A. (2023, March 18). Binance Portfolio Forecasting Hourly VWAP Dataset. https://doi.org/10.17605/OSF.IO/FJSUH

[3] Sbrana, A., Pires, G. (2023, March 18). Binance-VWAP-Dataset. GitHub, Zenodo. https://doi.org/10.5281/zenodo.7749449