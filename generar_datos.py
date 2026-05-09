import pandas as pd
import numpy as np
import os

np.random.seed(42)

clientes = []

for i in range(1, 121):

    if i <= 40:
        edad = np.random.randint(21, 35)
        ingreso_mensual = np.random.randint(1200000, 2800000)
        monto_credito = np.random.randint(30000000, 90000000)
        puntaje_crediticio = np.random.randint(450, 620)
        antiguedad_laboral = np.random.randint(1, 4)

    elif i <= 80:
        edad = np.random.randint(30, 48)
        ingreso_mensual = np.random.randint(2800000, 6000000)
        monto_credito = np.random.randint(90000000, 180000000)
        puntaje_crediticio = np.random.randint(620, 760)
        antiguedad_laboral = np.random.randint(3, 8)

    else:
        edad = np.random.randint(38, 60)
        ingreso_mensual = np.random.randint(6000000, 14000000)
        monto_credito = np.random.randint(180000000, 400000000)
        puntaje_crediticio = np.random.randint(760, 900)
        antiguedad_laboral = np.random.randint(7, 20)

    cuota_mensual = int(monto_credito / np.random.randint(180, 300))

    clientes.append({
        "id_cliente": i,
        "edad": edad,
        "ingreso_mensual": ingreso_mensual,
        "monto_credito": monto_credito,
        "puntaje_crediticio": puntaje_crediticio,
        "antiguedad_laboral": antiguedad_laboral,
        "cuota_mensual": cuota_mensual
    })

df = pd.DataFrame(clientes)

os.makedirs("data", exist_ok=True)

df.to_csv("data/clientes_credito.csv", index=False, encoding="utf-8")

print("Dataset generado correctamente en data/clientes_credito.csv")
print(df.head())
print("Total de registros:", len(df))