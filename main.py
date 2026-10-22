import sys
from plot_results import plot_fit
from src.solver import run_indometh_fit

def print_separator(title:str):
    print(f"\n{'-'*15} {title} {'-'*15}")

def main():
    print("Iniciando ajuste del modelo bicompartimental para Indometacina...")

    # esto por defecto
    subject_id = None

    try:
        results = run_indometh_fit(subject_id=subject_id, max_iter=100, tol=1e-6)
    except Exception as e:
        print(f"\nError durante la ejecucion del solver: {e}")
        sys.exit(1)

    # presentacion de los resultados
    print_separator("INFORMACION GENERAL")
    print(f"Sujeto analizado : {results['subject_id']}")
    print(f"Estado convergencia: {'Exito' if results['converged'] else 'fracaso'}")
    print(f"Iteraciones : {results['num_iter']}")

    print_separator("MACROPARAMETROS (Modelo Biexponencial)")
    print("Valores iniciales (Curve Stripping):")
    print(f"    A0  ={results['theta_init'][0]:.4f}")
    print(f"alpha0  ={results['theta_init'][1]:.4f}")
    print(f"    B0  ={results['theta_init'][2]:.4f}")
    print(f" beta0  ={results['theta_init'][3]:.4f}")

    print("\nParametros optimizados (Gauss-Newton):")
    print(f"    A = {results['theta_opt'][0]:.4f}")
    print(f"alpha = {results['theta_opt'][1]:.4f}")
    print(f"    B = {results['theta_opt'][2]:.4f}")
    print(f" beta = {results['theta_opt'][3]:.4f}")

    print_separator("METRICAS DE AJUSTE")
    print(f"suma de residuos al cuadrado (SSE): {results['sse']:>8.4f}")
    print(f"Coeficiente de Determinacion (R`2): {results['r_squared']:>8.4f}")

    print_separator("MICROPARAMETROS FARMACOCINETICOS")
    micro = results['micro_params']
    print(f"Concentracion inicial(C0): {micro['C0']:.4f} ug / mL")
    print(f"Volumen de dist. (V1) : {micro['V1']:.4f} L")
    print(f"Tasa de eliminacion (k10): {micro['k10']:.4f} h`-1")
    print(f"Tasa de distrib. (k12): {micro['k12']:.4f} h`-1")
    print(f"Tasa de retorno (k21): {micro['k21']:.4f} h`-1")

    print("\nProceso finalizado con exito./n")

    plot_fit(
        t=results['t'],
        y=results['y'],
        theta=results['theta_opt'],
        subject_id=str(results['subject_id'])
        )

if __name__ == "__main__":
    main()
