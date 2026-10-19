"""
comparacion.py — Comparación de configuraciones del modelo
Reporte T.1.3 de servicios fijos · CRC

Evalúa distintas decisiones de diseño —ponderación entre el componente
global y el segmentado, umbral de corte y criterio de segmentación— mediante
indicadores calculables sin etiquetas: estabilidad ante la aleatoriedad,
separación entre grupos, concordancia con el modelo global y contraste
frente a las reglas de validación del formato.

Ninguno de estos indicadores establece cuál configuración detecta
correctamente un mayor número de inconsistencias. Esa comparación requiere
casos de error conocido y se realiza en validacion.py.
"""

import numpy as np
import pandas as pd

from metricas import incumple_reglas, jaccard, _d_cohen


def _normalizar(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-12)


def comparar_ponderaciones(df, contaminacion, pesos=(0.0, 0.25, 0.5, 0.75, 1.0),
                           incumple=None):
    """
    Compara el resultado del modelo bajo distintas ponderaciones entre la
    puntuación global y la de los modelos por segmento.

    Requiere que df contenga SCORE_GLOBAL y SCORE_LOCAL.
    """
    if incumple is None:
        incumple = incumple_reglas(df)

    g = _normalizar(df["SCORE_GLOBAL"])
    l = _normalizar(df["SCORE_LOCAL"])

    # Referencia: conjunto señalado por el modelo puramente global
    u_g = g.quantile(1 - contaminacion)
    ref = set(np.where(g >= u_g)[0])

    filas = []
    for p in pesos:
        s = (1 - p) * g + p * l
        u = s.quantile(1 - contaminacion)
        marc = s >= u

        idx_marc = set(np.where(marc)[0])
        a, n = s[marc].to_numpy(), s[~marc].to_numpy()
        n_marc = int(marc.sum())
        n_cumple = int((marc & ~incumple).sum())

        filas.append({
            "Peso local": p,
            "Configuración": ("Solo global" if p == 0 else
                              "Solo segmentado" if p == 1 else
                              f"Combinado {int((1-p)*100)}/{int(p*100)}"),
            "Señalados": n_marc,
            "Separación (d de Cohen)": round(_d_cohen(a, n), 3),
            "Razón de medias": round(float(a.mean() / max(n.mean(), 1e-9)), 2),
            "Coincidencia con el global (Jaccard)": round(jaccard(idx_marc, ref), 4),
            "No detectables por las reglas": n_cumple,
            "% no detectable por las reglas": round(100 * n_cumple / max(n_marc, 1), 2),
        })

    return pd.DataFrame(filas)


def nucleo_comun_ponderaciones(df, contaminacion,
                              pesos=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """
    Registros señalados bajo todas las ponderaciones evaluadas.

    La ponderación equitativa entre ambos componentes no se deriva de un
    criterio de optimización, porque no existe una medida de desempeño que
    optimizar sobre los datos reales. Lo que sí puede establecerse es en qué
    medida el resultado depende de ese valor: el núcleo común identifica los
    registros que serían señalados con cualquiera de las ponderaciones, y la
    comparación con el conjunto adoptado dimensiona el margen que introduce
    la decisión.
    """
    g = _normalizar(df["SCORE_GLOBAL"])
    l = _normalizar(df["SCORE_LOCAL"])

    conjuntos = {}
    for p in pesos:
        s = (1 - p) * g + p * l
        u = s.quantile(1 - contaminacion)
        conjuntos[p] = set(np.where(s >= u)[0])

    nucleo = set.intersection(*conjuntos.values())
    union = set.union(*conjuntos.values())
    intermedios = [p for p in pesos if 0 < p < 1]
    nucleo_int = (set.intersection(*[conjuntos[p] for p in intermedios])
                  if intermedios else set())
    n_ref = len(conjuntos[0.5]) if 0.5 in conjuntos else len(next(iter(conjuntos.values())))

    return pd.DataFrame({
        "Indicador": [
            "Ponderaciones evaluadas",
            "Señalados por la configuración adoptada (50/50)",
            "Núcleo común a todas las ponderaciones",
            "Núcleo común sobre el conjunto adoptado (%)",
            "Núcleo común a las ponderaciones intermedias (25/50/75)",
            "Núcleo intermedio sobre el conjunto adoptado (%)",
            "Unión de todos los conjuntos señalados",
            "Registros que dependen de la ponderación elegida"],
        "Valor": [
            len(pesos),
            f"{n_ref:,}",
            f"{len(nucleo):,}",
            round(100 * len(nucleo) / max(n_ref, 1), 2),
            f"{len(nucleo_int):,}",
            round(100 * len(nucleo_int) / max(n_ref, 1), 2),
            f"{len(union):,}",
            f"{len(union) - len(nucleo):,}"]
    })


def comparar_umbrales(df, umbrales=(0.005, 0.01, 0.02, 0.05), incumple=None,
                      columna="SCORE_COMBINADO"):
    """
    Efecto del umbral de corte sobre el volumen y la composición de los
    registros señalados, manteniendo fija la configuración del modelo.

    Informa además cuántos registros del conjunto adoptado (1%) se conservan
    en cada umbral, lo que responde a cuántos registros cambian al modificar
    la proporción de contaminación.
    """
    if incumple is None:
        incumple = incumple_reglas(df)

    s = df[columna]
    base = set(np.where(s >= s.quantile(0.99))[0])

    filas = []
    for c in umbrales:
        u = s.quantile(1 - c)
        marc = s >= u
        idx = set(np.where(marc)[0])
        n_marc = int(marc.sum())
        n_cumple = int((marc & ~incumple).sum())
        filas.append({
            "Umbral de corte (%)": round(100 * c, 2),
            "Puntuación en el umbral": round(u, 4),
            "Señalados": n_marc,
            "Señalados por trimestre (promedio)":
                int(round(n_marc / df["ORDEN"].nunique())),
            "Coincidencia con el umbral del 1% (Jaccard)": round(jaccard(idx, base), 4),
            "Del conjunto del 1% se conservan": len(idx & base),
            "No detectables por las reglas": n_cumple,
            "% no detectable por las reglas": round(100 * n_cumple / max(n_marc, 1), 2),
        })
    return pd.DataFrame(filas)


def comparar_segmentaciones(df, X, detector, contaminacion, peso_local,
                            esquemas, min_por_segmento=200, incumple=None):
    """
    Compara distintos criterios de definición de los subespacios locales.

    esquemas : diccionario {nombre: [columnas]} con las variables que definen
               cada subespacio, por ejemplo
               {"Municipio": ["ID_MUNICIPIO"],
                "Departamento": ["ID_DEPARTAMENTO"],
                "Municipio y tecnología": ["ID_MUNICIPIO", "ID_TECNOLOGIA"]}

    Advertencia: reentrena todos los modelos locales de cada esquema, por lo
    que su costo computacional es elevado.
    """
    if incumple is None:
        incumple = incumple_reglas(df)

    g = _normalizar(df["SCORE_GLOBAL"])
    u_g = g.quantile(1 - contaminacion)
    ref = set(np.where(g >= u_g)[0])

    filas = []
    for nombre, cols in esquemas.items():
        clave = df[cols].astype(str).agg("|".join, axis=1)
        sc, n_mod, cob = detector.ajustar_local(X, clave, min_por_segmento,
                                                verbose=False)
        sc = sc.fillna(df["SCORE_GLOBAL"])
        s = (1 - peso_local) * g + peso_local * _normalizar(sc)
        u = s.quantile(1 - contaminacion)
        marc = s >= u
        n_marc = int(marc.sum())
        n_cumple = int((marc & ~incumple).sum())

        filas.append({
            "Criterio de segmentación": nombre,
            "Subespacios con modelo": n_mod,
            "Cobertura de registros (%)": round(cob, 1),
            "Señalados": n_marc,
            "Coincidencia con el global (Jaccard)":
                round(jaccard(set(np.where(marc)[0]), ref), 4),
            "% no detectable por las reglas": round(100 * n_cumple / max(n_marc, 1), 2),
        })
    return pd.DataFrame(filas)
