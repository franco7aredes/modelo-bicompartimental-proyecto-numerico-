import numpy as np
import matplotlib.pyplot as plt
from src.model import evaluate_model

def plot_fit(t: np.ndarray, y:np.ndarray, theta: np.ndarray, subject_id: str):
    '''
    grafica los datos experimentales y la curva ajustada
    del modelo biexponencial
    '''

    t_grid = np.linspace(0, max(t) * 1.05, 200)
    y_fit = evaluate_model(theta, t_grid)

    plt.figure(figsize=(8, 5))
    plt.scatter(t, y, color='red', label='Datos experimentales', zorder=5)
    plt.plot(t_grid, y_fit, color='blue', label='Ajuste Gauss-Newton', zorder=4)

    titulo =f"Ajuste cinetico - Indometacina (Sujeto: {subject_id})"
    plt.title(titulo)
    plt.xlabel("Tiempo (horas)")
    plt.ylabel("Concentracion plasmatica (mcg/ml)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    nombre_archivo = f"ajuste_indometacina_sujeto_{subject_id}.png"
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    plt.show()
