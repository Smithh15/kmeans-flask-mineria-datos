from flask import Flask, render_template, request
import pandas as pd
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

DATASET_PATH = "data/clientes_credito.csv"
IMG_PATH = "static/img"

os.makedirs(IMG_PATH, exist_ok=True)


def cargar_dataset():
    df = pd.read_csv(DATASET_PATH)
    return df


def preparar_datos(df):
    variables = [
        "ingreso_mensual",
        "monto_credito",
        "puntaje_crediticio",
        "cuota_mensual"
    ]

    datos = df[variables].copy()

    # Limpieza básica
    datos = datos.dropna()

    # Normalización
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(datos)

    return datos, datos_escalados, scaler, variables


def generar_grafica_codo(datos_escalados):
    inercias = []
    valores_k = range(1, 11)

    for k in valores_k:
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        modelo.fit(datos_escalados)
        inercias.append(modelo.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(valores_k, inercias, marker="o")
    plt.title("Método del codo")
    plt.xlabel("Número de clústeres K")
    plt.ylabel("Inercia")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{IMG_PATH}/metodo_codo.png")
    plt.close()


def ejecutar_kmeans(k):
    df = cargar_dataset()
    datos, datos_escalados, scaler, variables = preparar_datos(df)

    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = modelo.fit_predict(datos_escalados)

    df_resultado = df.copy()
    df_resultado["cluster"] = clusters

    # Centroides en escala original
    centroides_originales = scaler.inverse_transform(modelo.cluster_centers_)

    df_centroides = pd.DataFrame(
        centroides_originales,
        columns=variables
    )

    df_centroides["cluster"] = range(k)

    generar_grafica_codo(datos_escalados)
    generar_grafica_clusters(df_resultado, df_centroides)

    return df_resultado, df_centroides


def generar_grafica_clusters(df_resultado, df_centroides):
    plt.figure(figsize=(9, 6))

    for cluster in sorted(df_resultado["cluster"].unique()):
        datos_cluster = df_resultado[df_resultado["cluster"] == cluster]

        plt.scatter(
            datos_cluster["ingreso_mensual"],
            datos_cluster["monto_credito"],
            label=f"Clúster {cluster}",
            alpha=0.7
        )

    plt.scatter(
        df_centroides["ingreso_mensual"],
        df_centroides["monto_credito"],
        marker="X",
        s=250,
        color="black",
        label="Centroides"
    )

    plt.title("Visualización de clústeres y centroides")
    plt.xlabel("Ingreso mensual")
    plt.ylabel("Monto de crédito solicitado")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{IMG_PATH}/clusters.png")
    plt.close()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dataset")
def dataset():
    df = cargar_dataset()

    total_registros = df.shape[0]
    total_columnas = df.shape[1]
    columnas = df.columns.tolist()

    tabla_datos = df.head(20).to_html(
        classes="table",
        index=False
    )

    estadisticas = df.describe().to_html(
        classes="table",
        index=True
    )

    return render_template(
        "dataset.html",
        total_registros=total_registros,
        total_columnas=total_columnas,
        columnas=columnas,
        tabla_datos=tabla_datos,
        estadisticas=estadisticas
    )


@app.route("/kmeans")
def kmeans():
    return render_template("kmeans.html")


@app.route("/resultados")
def resultados():
    k = request.args.get("k", default=3, type=int)

    if k < 2:
        k = 2

    if k > 10:
        k = 10

    df_resultado, df_centroides = ejecutar_kmeans(k)

    tabla_resultados = df_resultado.head(30).to_html(
        classes="table",
        index=False
    )

    tabla_centroides = df_centroides.round(2).to_html(
        classes="table",
        index=False
    )

    resumen_clusters = df_resultado.groupby("cluster")[
        [
            "edad",
            "ingreso_mensual",
            "monto_credito",
            "puntaje_crediticio",
            "antiguedad_laboral",
            "cuota_mensual"
        ]
    ].mean().round(2)

    resumen_clusters["cantidad_clientes"] = df_resultado.groupby("cluster").size()

    tabla_resumen = resumen_clusters.to_html(
        classes="table",
        index=True
    )

    version = int(time.time())

    return render_template(
        "resultados.html",
        k=k,
        tabla_resultados=tabla_resultados,
        tabla_centroides=tabla_centroides,
        tabla_resumen=tabla_resumen,
        version=version
    )


@app.route("/interpretacion")
def interpretacion():
    return render_template("interpretacion.html")


if __name__ == "__main__":
    app.run(debug=True)