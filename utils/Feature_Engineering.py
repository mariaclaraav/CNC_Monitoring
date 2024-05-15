import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.signal import lfilter
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.auto import tqdm
from PyEMD import EMD
import pywt
from sklearn.metrics import mean_squared_error

from nitime.algorithms.autoregressive import AR_est_YW
#from scipy.signal import rfft




def statistical_features(data, window_size, step_size):

    rolling_mean = data.rolling(window=window_size, step=step_size).mean()
    rolling_std = data.rolling(window=window_size, step=step_size).std()
    rolling_max = data.rolling(window=window_size, step=step_size).max()
    rolling_min = data.rolling(window=window_size, step=step_size).min()
    rolling_rms = data.rolling(window=window_size, step=step_size).apply(lambda x: np.sqrt(np.sum(x ** 2) / len(x)), raw=True)
    rolling_median = data.rolling(window=window_size, step=step_size).median()    
    rolling_variance = data.rolling(window=window_size, step=step_size).var()
    rolling_skewness = data.rolling(window=window_size, step=step_size).skew()
    rolling_kurtosis = data.rolling(window=window_size, step=step_size).kurt()


    #Non-dimensional
    #rolling_shape_factor = data.rolling(window=window_size, step=step_size).apply(lambda x: np.sqrt(np.sum(x ** 2) / (np.sum(x) / len(x))), raw=True)
    #rolling_crest_factor = data.rolling(window=window_size, step=step_size).apply(lambda x: np.max(np.abs(x)) / np.sqrt(np.sum(x ** 2) / len(x)), raw=True)
    rolling_impulse_factor = data.rolling(window=window_size, step=step_size).apply(lambda x: np.max(np.abs(x)) / (np.sum(np.abs(x)) / len(x)), raw=True)
    rolling_margin_factor = data.rolling(window=window_size, step=step_size).apply(lambda x: np.max(np.abs(x)) / ((np.sum(np.sqrt(np.abs(x))) / len(x)) ** 2), raw=True)
  

    # Concatenate all features into a DataFrame
    df = pd.concat([data, rolling_mean, rolling_std, rolling_max, rolling_min,
                    rolling_median, rolling_variance, rolling_skewness,
                    rolling_kurtosis, rolling_rms,rolling_impulse_factor, rolling_margin_factor], axis=1, ignore_index=True)

    # Naming the columns
    df.columns = [data.name, 'Rolling Mean', 'Rolling Std', 'Rolling Max',
                  'Rolling Min', 'Rolling Median', 'Rolling Variance',
                  'Rolling Skewness', 'Rolling Kurtosis', 'Rolling RMS', 'Rolling Impulse Factor', 'Rolling Margin Factor']

    # Remove rows with NaN values
    df = df.dropna()

    return df

def energy_features(data, window_size, step_size):
    """
    Calculates energy-based features over a rolling window for given data.

    Parameters:
    data (pd.Series): The input data series.
    window_size (int): The size of the rolling window.

    Returns:
    pd.DataFrame: A DataFrame containing the original data and calculated energy features.
    """

    # Calculate the total energy within each window, which is the sum of the squares of the elements
    rolling_energy = data.rolling(window=window_size, step=step_size).apply(lambda x: np.sum(x**2), raw=True)
    
    # Define a function to calculate energy entropy for each window
    def entropy_of_energy(x):
        energy = x**2  # Squaring the values to get the energy
        total_energy = np.sum(energy)  # Total energy in the window
        if total_energy > 0:
            energy_probability = energy / total_energy  # Probability distribution of energy
            # Calculate entropy using the probability distribution, add epsilon to avoid log(0)
            return -np.sum(energy_probability * np.log2(energy_probability + np.finfo(float).eps))
        else:
            return 0  # Entropy is zero if total energy is zero

    # Apply the entropy calculation to each rolling window
    rolling_entropy = data.rolling(window=window_size, step=step_size).apply(entropy_of_energy, raw=True)

    # Calculate normalized energy as the total energy normalized by the number of samples in the window
    rolling_normalized_energy = data.rolling(window=window_size, step=step_size).apply(lambda x: np.sum(x**2) / len(x), raw=True)

    # Concatenate all features into a DataFrame for easy analysis
    df = pd.concat([data, rolling_energy, rolling_entropy, rolling_normalized_energy], axis=1)

    # Naming the columns to reflect the calculated features
    df.columns = [data.name, 'Rolling Energy', 'Rolling Energy Entropy', 'Rolling Normalized Energy']

    # Remove rows with NaN values that result from the rolling calculation at the start of the series
    df = df.dropna()

    return df

def frequency_domain_features(data, window_size, step_size):
    """Calculate frequency domain features from a time series data."""

        # Frequency domain features
    def compute_power_spectrum(x, normalize = True):
        n = len(x)  # Tamanho do sinal
        # Realiza a FFT
        #freq_data = rfft(x) # com o scipy mas ta dando erro
        freq_data = np.fft.rfft(x)
        
        # Normaliza a FFT pelo número de pontos no sinal
        if normalize:
            freq_data = freq_data / n
        
        # Calcula o espectro de potência
        power_spectrum = np.abs(freq_data) ** 2
        
        # Remover o componente de frequência zero
        mask = np.ones_like(power_spectrum, dtype=bool)
        mask[0] = False
        return power_spectrum[mask]

    # Calculate spectral mean, std, and kurtosis for each rolling window
    rolling_mean = data.rolling(window=window_size, step=step_size).apply(lambda x: np.mean(compute_power_spectrum(x)), raw=True)
    rolling_std = data.rolling(window=window_size, step=step_size).apply(lambda x: np.std(compute_power_spectrum(x)), raw=True)
    rolling_kurtosis = data.rolling(window=window_size, step=step_size).apply(lambda x: kurtosis(compute_power_spectrum(x)), raw=True)

    # You can define other frequency domain features here

    # Concatenate all features into a DataFrame
    freq_features = pd.concat([data, rolling_mean, rolling_std, rolling_kurtosis], axis=1)

    # Naming the columns
    freq_features.columns = [data.name, 'Spectral Rolling Mean', 'Spectral Rolling Std', 'Spectral Rolling Kurtosis']

    # Remove rows with NaN values
    freq_features = freq_features.dropna()

    return freq_features

def DWT_decomp(data, w = 'db14', level=3, figsize=(30, 30), dt=1/2000, plot = False):
    """
    Decompor e plotar um sinal usando a decomposição wavelet.
    
    S = An + Dn + Dn-1 + ... + D1
    
    Parâmetros:
        data (array-like): Dados do sinal de entrada.
        w (str): Nome da wavelet a ser usada para a decomposição.
        title (str): Título para o gráfico.
        level (int, opcional): Nível de decomposição. O padrão é 5.
        figsize (tuple, opcional): Tamanho da figura em polegadas (largura, altura). O padrão é (30, 30).
        dt (float, opcional): Intervalo de amostragem do sinal em segundos.

    Retorna:
        df_ca (DataFrame): DataFrame contendo os coeficientes de aproximação.
        df_cd (DataFrame): DataFrame contendo os coeficientes de detalhe.
        df_coef (DataFrame): DataFrame contendo as colunas desejadas.
    """
 

    # Verificar se o tamanho dos dados é ímpar e ajustar, se necessário
    if len(data) % 2 != 0:
        data = data[:-1]
        
    # Criar o objeto Wavelet
    w = pywt.Wavelet(w)
    a = data
    ca = []  # Coeficientes de aproximação
    cd = []  # Coeficientes de detalhe
    
    # Realizar a decomposição wavelet
    for i in range(level):
        (a, d) = pywt.dwt(a, w)
        ca.append(a)
        cd.append(d)

    # Determinar o tamanho final dos coeficientes
    min_length = min(len(c) for c in ca)
    
    # Recriar os sinais de aproximação e detalhe
    rec_a = []
    rec_d = []

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w)) # transformada wavelet inversa

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w))

    if plot: 
        # Plotar o sinal original e os sinais decompostos
        fig = plt.figure(figsize=figsize)
        ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
        ax_main.set_title(data.name)
        ax_main.plot(data.values)
        ax_main.set_xlim(0, len(data) - 1)
        
        for i, y in enumerate(rec_a):
            ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
            ax.plot(y, 'r')
            ax.set_xlim(0, len(y) - 1)
            ax.set_ylabel("A{}".format(i + 1))
        
        for i, y in enumerate(rec_d):
            ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
            ax.plot(y, 'g')
            ax.set_xlim(0, len(y) - 1)
            ax.set_ylabel("D{}".format(i + 1))

        plt.tight_layout()
        plt.show()
        
    # Últimas colunas dos DataFrames
    last_ca = pd.Series(rec_a[-1])
    last_cd = pd.DataFrame(rec_d).T

    # Criar DataFrames para os coeficientes
    df_ca = pd.DataFrame(rec_a).T
    df_cd = pd.DataFrame(rec_d).T

    # Renomear as colunas dos DataFrames df_ca e df_cd
    df_ca = df_ca.rename(columns={col: f'A{col + 1}' for col in df_ca.columns})
    df_cd = df_cd.rename(columns={col: f'D{col + 1}' for col in df_cd.columns})

    # Criar df_coef combinando as últimas colunas de df_ca e df_cd
    df_coef = pd.concat([df_cd, last_ca], axis=1)

    # Renomear a última coluna de df_coef para corresponder ao nome original de df_ca
    df_coef = df_coef.rename(columns={df_coef.columns[-1]: df_ca.columns[-1]})

    #Removendo as linhas com Nan que podem aparecer 
    df_coef.dropna(axis=0,inplace=True)

    # Somar todas as colunas para reconstruir o sinal
    
    soma_colunas_nan = df_coef.sum(axis=1)

    residuo = (data.values - soma_colunas_nan.values)

    #column_names = data.name.tolist() + df_coef.columns.tolist()

    #df_final = pd.concat([data,df_coef], axis=1, ignore_index=True)
    
    #df_final.columns = [data.name, 'D1','D2','D3','A3']

    # Concatenar os DataFrames com o DataFrame original e ajustar o índice
    df_final = pd.concat([data.reset_index(drop=True), df_coef], axis=1)

    # Renomear as colunas
    df_final.columns = [data.name] + [f'D{i}' for i in range(1, len(df_coef.columns))] + [df_ca.columns[-1]]
    
    if plot: 
        plt.figure(figsize=(15,3))
        plt.plot(residuo)
        plt.title("Resíduo")
        plt.xlabel("Amostra")
        plt.ylabel("Valor")
        plt.tight_layout()
        plt.show()

    return df_final

def rolling_approximate_entropy(data, window_size, m, r, step_size):
    def entropy_approximate(window):
        if window.dropna().shape[0] < m:  # Checa se há dados suficientes na janela
            return np.nan  # Retorna NaN se não houver dados suficientes
        
        def _max_distance(sub_i, sub_j):
            return max(abs(sub_i - sub_j))
        
        def _phi(m):
            N = len(window)
            C = np.zeros((N - m + 1,))
            for i in range(N - m + 1):
                x_i = window[i:i + m]
                count = 0
                for j in range(N - m + 1):
                    if i != j:
                        x_j = window[j:j + m]
                        if _max_distance(x_i, x_j) <= r:
                            count += 1
                C[i] = count / (N - m + 1)
            if np.all(C == 0):  # Evita log(0)
                return 0
            return np.sum(np.log(C[C > 0])) / (N - m + 1)
        
        return abs(_phi(m) - _phi(m + 1))
    
    # Calculando a entropia aproximada de forma rolante
    rolling_entropy = data.rolling(window=window_size, step=step_size).apply(entropy_approximate, raw=True)

    # Concatenate all features into a DataFrame
    df_ApEn = pd.concat([data, rolling_entropy], axis=1)
    

    # Naming the columns
    df_ApEn.columns = [data.name, 'Rolling ApEn']

    # Remove rows with NaN values
    df_ApEn = df_ApEn.dropna()

    return df_ApEn

def calculate_jerk(data, fs=2000):

    dt = 1/fs
    dv = np.diff(data)

    # Calculate the jerk (dv / dt)
    jerk = dv / dt

    jerk = pd.DataFrame(jerk)

    df_final = pd.concat([data.reset_index(drop=True), jerk], axis=1)

    df_final.columns = [data.name, 'Jerk']

    return df_final  

def process_time_series(df, feature_type='statistical', window_size=500, sampling_rate=2000, step_size = None,  m=None, r=None):
    """
    Process a DataFrame with time series data to extract specified features.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing time series data.
        feature_type (str): Type of features to extract ('statistical', 'energy', 'frequency', or 'wavelet').
        window_size (int): Window size for rolling operations.
        sampling_rate (int): Sampling rate of the data (required for 'frequency' type).

    Returns:
        pd.DataFrame: DataFrame containing the processed time series features.
    """
    
    # Defining the additional columns to keep
    additional_columns = ['Time', 'Month', 'Year', 'Machine', 'Process', 'Label', 'Unique_Code', 'Period']
    
    # List to store the processed dataframes
    processed_dfs = []
    
    # Define a dictionary to select feature function based on type
    feature_functions = {
        'statistical': statistical_features,
        'energy': energy_features,
        'frequency': frequency_domain_features,
        'wavelet': DWT_decomp,
        'ApEn': rolling_approximate_entropy,
        'jerk': calculate_jerk,
    }
    
    # Check if a valid feature function has been chosen
    if feature_type not in feature_functions:
        raise ValueError(f"Feature type {feature_type} is not recognized. Choose 'statistical', 'energy', 'frequency', or 'wavelet'.")

    feature_function = feature_functions[feature_type]
    
    if feature_type in ['wavelet', 'jerk']:
        for unique_code, group in tqdm(df.groupby('Unique_Code'), desc="Processing Time Series"):
            # Apply the chosen feature function for X, Y, Z
            features_X = feature_function(data=group['X_axis'])
            features_Y = feature_function(data=group['Y_axis'])
            features_Z = feature_function(data=group['Z_axis'])

            # Rename the columns to indicate the axis
            features_X.columns = ['X_' + col for col in features_X.columns]
            features_Y.columns = ['Y_' + col for col in features_Y.columns]
            features_Z.columns = ['Z_' + col for col in features_Z.columns]
            # Select and reset the index of the additional columns
            additional_data = group[additional_columns]  # No need to adjust window_size
            
            # Concatenate the features with the additional columns
            final_df = pd.concat([additional_data.reset_index(drop=True), features_X.reset_index(drop=True), features_Y.reset_index(drop=True), features_Z.reset_index(drop=True)], axis=1)
                
            # Remove rows with NaN resulting from rolling operations (if any)
            final_df = final_df.dropna()
                
            # Add the processed DataFrame to the list
            processed_dfs.append(final_df)

    else:
        for unique_code, group in tqdm(df.groupby('Unique_Code'), desc="Processing Time Series"):
            if feature_type == 'ApEn':
                # Specific call for Approximate Entropy
                features_X = feature_function(data=group['X_axis'], window_size=window_size, step_size=step_size, m=m, r=r)
                features_Y = feature_function(data=group['Y_axis'], window_size=window_size, step_size=step_size, m=m, r=r)
                features_Z = feature_function(data=group['Z_axis'], window_size=window_size, step_size=step_size, m=m, r=r)
            else:
                # Apply the chosen feature function for X, Y, Z
                features_X = feature_function(data=group['X_axis'], window_size=window_size, step_size=step_size)
                features_Y = feature_function(data=group['Y_axis'], window_size=window_size, step_size=step_size)
                features_Z = feature_function(data=group['Z_axis'], window_size=window_size, step_size=step_size)
            
            # Rename the columns to indicate the axis
            features_X.columns = ['X_' + col for col in features_X.columns]
            features_Y.columns = ['Y_' + col for col in features_Y.columns]
            features_Z.columns = ['Z_' + col for col in features_Z.columns]
            
            # Select and reset the index of the additional columns
            additional_data = group[additional_columns].iloc[window_size-1:]  # Adjust to match the size after rolling operations
            
            # Concatenate the features with the additional columns
            final_df = pd.concat([additional_data.reset_index(drop=True), features_X.reset_index(drop=True), features_Y.reset_index(drop=True), features_Z.reset_index(drop=True)], axis=1)
            
            # Remove rows with NaN resulting from rolling operations
            final_df = final_df.dropna()
            
            # Add the processed DataFrame to the list
            processed_dfs.append(final_df)   
        
    # Concatenate all processed DataFrames into a single DataFrame
    final_result = pd.concat(processed_dfs)
    
    final_result['Period_Num'] = pd.to_datetime(final_result['Period'], format='%b-%Y')
    final_result.sort_values(by=['Period_Num', 'Unique_Code', 'Time'], inplace=True)
    final_result.drop(['Period_Num'], axis=1, inplace=True)
    final_result.reset_index(drop=True,inplace=True)
    
    return final_result