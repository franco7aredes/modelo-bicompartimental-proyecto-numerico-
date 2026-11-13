#!/bin/bash

if [ ! -d "venv" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv venv

    echo "Activando entorno virtual..."
    source venv/bin/activate

    echo "Instalando dependencias desde requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt

else
    echo "El entorno virtual ya existe. Activando..."
    source venv/bin/activate
fi

echo "======================================="
echo "Entorno listo  -> Ejecutando el proyecto..."
echo "======================================="
python3 main.py
