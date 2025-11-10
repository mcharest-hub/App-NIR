import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
from sklearn.preprocessing import MinMaxScaler


# SNV Normalisation

def apply_snv(df):
    """
    Apply Standard Normal Variate (SNV) transformation to each row of DataFrame.
    This standardizes each row to have zero mean and unit variance.

    Args:
        df (pd.DataFrame): DataFrame to transform

    Returns:
        pd.DataFrame: SNV transformed Dataframe
    """
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1, ddof=0), axis=0)

# Savitzky-Golay filter/derivative

def apply_savgol(df, window_length, polyorder, **kwargs):
    """
    Apply Savitzky-Golay filter to smooth data in a DataFrame

    Args:
        df (pd.DataFrame): DataFrame containing data to smooth
        window_length (int): The length of the filter window (number coefficients). Must be a positive odd integer.
        polyorder (int): The order of the the polynomial used to fit the samples. Must be less than window_length.
        **kwargs : Additional keyword arguments to pass to savgol_filter (derivative for example)

    Returns:
        pd.DataFrame: Dataframe with Savitzky-Golay filter applied
    """
    return pd.DataFrame(savgol_filter(df.values, window_length, polyorder, **kwargs), columns=df.columns)

# Baseline correction (Rubberband)

def rubberband(x, y):
    """
    Compute the baseline of a spectrum using the rubberband method.
    
    This function finds the convex hull of the given spectrum and 
    interpolates a baseline using the convex hull vertices.

    Args:
        x (np.ndarray): The wavelengths (independent variable).
        y (np.ndarray): The intensities (dependent variable).

    Returns:
        np.ndarray: The computed baseline of the spectrum.
        
    """
    v = ConvexHull(np.array(list(zip(x, y)))).vertices
    v = np.roll(v, -v.argmin()) 
    v = v[:v.argmax()+1]   
   
    return np.interp(x, x[v], y[v])

# Function to apply Rubberband baseline correction to all spectra of the Dataframe
def correct_baseline(df):
    """
    Apply Rubber Band correction to each spectrum in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing raw Raman spectra. 
                                   Columns should represent wavelengths, and rows represent spectra.

    Returns:
        pd.DataFrame: DataFrame containing the corrected Raman spectra.
    """
    wavelengths = np.array(df.columns).astype(int)  # Wavelengths are the column names
    raman_corrected = pd.DataFrame(columns=wavelengths)

    # Apply Rubber Band correction to each spectrum in the DataFrame
    for index, spectrum in df.iterrows():
        # Apply the rubberband function to the spectrum and subtract the rubberband baseline from the spectrum
        rubberband_baseline = rubberband(wavelengths, spectrum.values)
        corrected_spectrum = spectrum.values - rubberband_baseline
        
        # Add the corrected spectrum to the raman_corrected DataFrame
        raman_corrected.loc[index] = corrected_spectrum

    return raman_corrected

# Min-Max Normalisation
def apply_minmax_scaler(df):
    """
    Apply MinMax scaling to the columns of a DataFrame.
    
    This function transposes the DataFrame, fits a MinMaxScaler to the data,
    and transforms it to scale the values between 0 and 1.

    Args:
    df (pd.DataFrame): Input DataFrame with numerical columns to scale.

    Returns:
    pd.DataFrame: DataFrame with MinMax scaled values.
    
    """
    scaler = MinMaxScaler()
    df_transposed = df.T
    scaled_data = scaler.fit_transform(df_transposed)
    scaled_df = pd.DataFrame(scaled_data, columns=df_transposed.columns)
    return scaled_df.T

# L2 norm normalization
def apply_l2_norm(df):
    """
    Apply L2 normalization to each row of the DataFrame.
    This scales each row to have an L2 norm (Euclidean norm) of 1.

    Args:
        df (pd.DataFrame): DataFrame to normalize

    Returns:
        pd.DataFrame: L2 normalized DataFrame
    """
    # Compute the L2 norm for each row
    l2_norm = np.sqrt((df ** 2).sum(axis=1))
    
    # Divide each element by the L2 norm
    return df.div(l2_norm, axis=0)

# Average spectra

def calculate_average_spectra(dataframe, sample_column='Sample'):
    """
    Process a spectral data DataFrame to compute average spectra by sample
    and concatenate it with aggregated metadata.
    """
    # Identify wavelength columns (assuming they are of type float)
    wavelength_columns = dataframe.select_dtypes(include=['float64']).columns.tolist()
    
    # Identify ignored columns (metadata, not of type float)
    ignored_columns = [col for col in dataframe.columns if col not in wavelength_columns]
    
    # Compute average spectra by sample using only the wavelength columns
    average_spectra_by_sample = dataframe.groupby(sample_column)[wavelength_columns].mean()
    
    # Aggregate metadata columns, keeping the first occurrence per sample
    ignored_data_aggregated = dataframe[ignored_columns].groupby(sample_column).first()
    
    # Concatenate aggregated metadata with the average spectra
    average_spectra = pd.concat([ignored_data_aggregated, average_spectra_by_sample], axis=1)
    
    # Reset the index to make the sample column a regular column
    average_spectra = average_spectra.reset_index()
    
    return average_spectra