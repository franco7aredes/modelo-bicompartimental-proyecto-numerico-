# modelo-bicompartimental-proyecto-numerico-
Estimación de Parámetros Farmacocinéticos en Modelos Bicompartimentales (proyecto de análisis numérico 2)

📄 **Informe del proyecto (en LaTeX):** [Ver en Overleaf](https://www.overleaf.com/project/6adea57d99fc2405737d34cc/share#05cef6bc9b4ff146f014e93de4ffb8c64101a43e5d8b1121)

---

## 🚀 Cómo ejecutar el proyecto

La forma más sencilla de inicializar el entorno y correr el orquestador principal es a través del script de configuración incluido, el cual crea el entorno virtual, instala las dependencias y ejecuta el programa automáticamente.

Desde la raíz del proyecto, abrí una terminal y ejecutá:

```bash
chmod +x setup.sh
./setup.sh
```

Alternativamente, si ya tenés el entorno virtual activado, podés ejecutar el flujo principal directamente con:
```bash
python3 main.py
```

---

## 🔍 Ejecución de módulos independientes (Diagnóstico)

El proyecto está diseñado bajo una arquitectura modular. Además del flujo principal, podés ejecutar módulos específicos de forma independiente para realizar diagnósticos e inspecciones sobre los datos y la estabilidad numérica.

Para ejecutar estos módulos, asegurate de estar en la raíz del proyecto y utilizar la bandera `-m` para que Python reconozca el paquete `src`:

### 1. Análisis de Estabilidad (Jacobiano y Ecuaciones Normales)
Ejecutando el modelo de manera aislada, el programa calculará y reportará los números de condición $\kappa(J)$ y $\kappa(J^T J)$ evaluados en la estimación inicial generada por *curve stripping*.
```bash
python3 -m src.model
```

### 2. Inspección del Dataset
Ejecutando el cargador de datos, podrás visualizar la estructura cruda de la base de datos *Indometh* (obtenida desde statsmodels) y verificar cómo se procesan y formatean los vectores temporales y de concentración poblacional.
```bash
python3 -m src.data_loader
```
