import statsmodels.api as sm
import pandas as pd
import numpy as np
from typing import Tuple, Optional

def fetch_raw_indometh() -> pd.Dataframe:
    '''
    Descarga e importa el dataset 'Indometh' desde statsmodels
    '''
    df = sm.datasets.get_rdataset("Indometh", "datasets").data
    return df

def get_indometh_data(subject_id: Optional[int] = None)
                    -> Tuple[np.ndarray, np.ndarray]:
    '''
    Procesa y formatea los datos de indometacina para ser utilizados por
    el solver
    subject_id: numero de sujeto del 1 al 6.
        si es None, calcula el promedio poblacional de los 6 sujetos
    retorna:
        t(np.ndarray): vector de tiempos (m,)
        y(np.ndarray): vector de concentraciones plasmaticas (m,)
    '''
    df = fetch_raw_indometh()
    if subject_id is not None:
        df_filtered = df[df['Subject'] == str(subject_id)].sort_values('time')
        t = df_filtered['time'].to_numpy(dtype=np.float64)
        y = df_filtered['conc'].to_numpy(dtype=np.float64)
    else:
        df_avg = df.groupby('time')['conc'].mean().reset_index()
        t = df_avg['time'].to_numpy(dtype=np.float64)
        y = df_avg['conc'].to_numpy(dtype=np.float64)

    return t, y
