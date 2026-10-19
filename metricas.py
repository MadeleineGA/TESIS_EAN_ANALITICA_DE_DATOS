"""
metricas.py — Evaluación del modelo de detección de anomalías
Reporte T.1.3 de servicios fijos · CRC

En ausencia de un conjunto de registros previamente etiquetados como
correctos o erróneos, las medidas de desempeño supervisado (precisión,
exhaustividad, F1) no son calculables sobre los datos reales. Para
obtenerlas se recurre a la inyección controlada de anomalías sintéticas,
implementada en validacion.py. Este módulo reúne los indicadores que sí
pueden calcularse sobre el reporte tal como fue entregado:

  1. Caracterización de la puntuación y separación entre grupos
  2. Estabilidad del modelo ante la aleatoriedad del entrenamiento
  3. Concordancia entre el componente global y el segmentado
  4. Estabilidad y concentración de la atribución por variable
  5. Contraste frente a las reglas de validación de primer nivel del formato
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, hypergeom
from sklearn.ensemble import IsolationForest

COLUMNAS_PUNTUACION = [("Global", "SCORE_GLOBAL"),
                       ("Local (por municipio)", "SCORE_LOCAL"),
                       ("Combinada", "SCORE_COMBINADO")]


# ============================================================================
# 1. CARACTERIZACIÓN DE LA PUNTUACIÓN
# ============================================================================
def descriptivos_puntuacion(df, contaminacion):
    """Estadísticos de las tres puntuaciones y su umbral de corte."""
    filas = []
    for nombre, col in COLUMNAS_PUNTUACION:
        s = df[col]
        filas.append({
            "Puntuación": nombre,
            "Mínimo": round(s.min(), 4),
            "P25": round(s.quantile(0.25), 4),
            "Mediana": round(s.median(), 4),
            "P75": round(s.quantile(0.75), 4),
            "P99": round(s.quantile(0.99), 4),
            "Máximo": round(s.max(), 4),
            "Desviación": round(s.std(), 4),
            "Umbral de corte": round(s.quantile(1 - contaminacion), 4),
        })
    return pd.DataFrame(filas)


def _d_cohen(a, n):
    """Diferencia estandarizada entre dos grupos (desviación combinada)."""
    sp = np.sqrt(((len(a) - 1) * np.var(a, ddof=1)
                  + (len(n) - 1) * np.var(n, ddof=1))
                 / max(len(a) + len(n) - 2, 1))
    return (np.mean(a) - np.mean(n)) / max(sp, 1e-12)


def _d_de_un_corte(valores, contaminacion):
    """d de Cohen entre las dos partes en que un cuantil divide una muestra."""
    u = np.quantile(valores, 1 - contaminacion)
    return _d_cohen(valores[valores >= u], valores[valores < u])


def referencia_separacion(contaminacion, n=200000, seed=42):
    """
    Valor de la d de Cohen que produce el corte por sí mismo.

    Cortar cualquier distribución continua unimodal en su percentil
    (1-contaminación) genera una diferencia estandarizada elevada por
    construcción, con independencia de que existan o no dos poblaciones.
    Esta función estima ese valor de referencia sobre distribuciones
    unimodales sin anomalías, de modo que la d observada en el modelo se
    interprete frente a él y no frente al umbral convencional de 0,8.

    Devuelve un DataFrame con la referencia por distribución.
    """
    rng = np.random.default_rng(seed)
    muestras = {
        "Normal": rng.standard_normal(n),
        "Beta (2, 8)": rng.beta(2, 8, n),
        "Log-normal": rng.lognormal(0, 1, n),
        "Exponencial": rng.exponential(1, n),
    }
    filas = [{"Distribución de referencia": k,
              "d de Cohen inducida por el corte": round(_d_de_un_corte(v, contaminacion), 3)}
             for k, v in muestras.items()]
    return pd.DataFrame(filas)


def separacion(df, contaminacion, seed=42, n_ref=200000):
    """
    Separación entre los registros señalados y el resto.

    La diferencia estandarizada (d de Cohen) expresa cuántas desviaciones
    típicas separan la puntuación media de ambos grupos. El valor se
    interpreta frente a la referencia nula —la d que el propio corte induce
    al dividir una distribución unimodal en el mismo percentil— y no frente
    al umbral convencional de 0,8, que corresponde a la comparación de dos
    grupos definidos con independencia de la variable medida. Un cociente
    próximo a la unidad indica que la separación observada es la que el
    corte produce por sí mismo; valores superiores indican una cola más
    pesada que la de una distribución unimodal.
    """
    rng = np.random.default_rng(seed)
    d_ref = _d_de_un_corte(rng.standard_normal(n_ref), contaminacion)

    filas = []
    for nombre, col in COLUMNAS_PUNTUACION:
        s = df[col].to_numpy()
        u = np.quantile(s, 1 - contaminacion)
        a, n = s[s >= u], s[s < u]
        d = _d_cohen(a, n)
        filas.append({
            "Puntuación": nombre,
            "Media señalados": round(float(np.mean(a)), 4),
            "Media resto": round(float(np.mean(n)), 4),
            "Razón de medias": round(float(np.mean(a) / max(np.mean(n), 1e-9)), 3),
            "d de Cohen": round(d, 3),
            "d de referencia (corte sobre normal)": round(d_ref, 3),
            "Cociente observado/referencia": round(d / max(d_ref, 1e-12), 2),
        })
    return pd.DataFrame(filas)


def forma_de_la_cola(df, contaminacion, columna="SCORE_COMBINADO"):
    """
    Indicadores de la forma de la distribución en el entorno del umbral.

    Permite establecer si el conjunto señalado constituye un grupo
    desprendido de la distribución —en cuyo caso existiría un salto entre el
    último registro no señalado y el primero señalado— o el extremo de un
    continuo. Se comparan la distancia entre registros contiguos alrededor
    del umbral con la distancia típica en el centro de la distribución.
    """
    s = np.sort(df[columna].to_numpy())
    n = len(s)
    k = int(np.ceil(contaminacion * n))
    corte = n - k

    dif = np.diff(s)
    salto_umbral = float(dif[corte - 1]) if 0 < corte < n else np.nan
    dif_centro = dif[int(0.45 * n):int(0.55 * n)]
    paso_central = float(np.median(dif_centro)) if len(dif_centro) else np.nan

    # Referencia pertinente: la distancia típica entre registros contiguos en
    # la propia cola. En el centro de la distribución la densidad es mucho
    # mayor, de modo que cualquier separación de la cola resulta elevada
    # frente a ella y la comparación no informaría sobre la existencia de una
    # discontinuidad.
    ini_cola = int(0.98 * n)
    dif_cola = dif[ini_cola:]
    paso_cola = float(np.median(dif_cola)) if len(dif_cola) else np.nan

    def f(v, d=8):
        return "—" if v is None or np.isnan(v) else f"{v:.{d}f}".rstrip("0").rstrip(".")

    return pd.DataFrame({
        "Indicador": [
            "Registros señalados",
            "Puntuación del último registro no señalado",
            "Puntuación del primer registro señalado",
            "Salto entre ambos",
            "Distancia típica entre contiguos en el 2% superior",
            "Razón entre el salto y la distancia típica de la cola",
            "Distancia típica entre contiguos en la zona central",
            "Razón entre el salto y la distancia típica central",
            "Registros en el intervalo de ±1% del umbral"],
        "Valor": [
            f"{k:,}",
            f(float(s[corte - 1]), 6),
            f(float(s[corte]), 6),
            f(salto_umbral),
            f(paso_cola),
            f(salto_umbral / max(paso_cola, 1e-12), 2),
            f(paso_central),
            f(salto_umbral / max(paso_central, 1e-12), 2),
            f"{int(((s >= s[corte] * 0.99) & (s <= s[corte] * 1.01)).sum()):,}"]
    })


# ============================================================================
# 2. ESTABILIDAD ANTE LA ALEATORIEDAD
# ============================================================================
def jaccard(a, b):
    """Índice de Jaccard entre dos conjuntos de índices."""
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)


# Alias conservado por compatibilidad con los módulos que lo importaban.
_jaccard = jaccard


def estabilidad_semillas(X, contaminacion, n_arboles=400, max_samples=256,
                         semillas=(42, 101, 202, 303, 404), n_muestra=300000,
                         rng=None):
    """
    Reentrena el modelo con distintas semillas y mide la coincidencia entre
    los conjuntos de registros señalados, mediante el índice de Jaccard.

    Un valor cercano a 1 indica que el resultado no depende de la
    aleatoriedad del particionamiento. Devuelve además las puntuaciones de
    cada repetición, necesarias para el análisis de validez equivalente.
    """
    rng = rng or np.random.default_rng(0)
    idx = (rng.choice(len(X), min(n_muestra, len(X)), replace=False)
           if len(X) > n_muestra else np.arange(len(X)))
    Xs = X.values[idx]

    conjuntos, umbrales, puntuaciones = [], [], []
    for s in semillas:
        m = IsolationForest(n_estimators=n_arboles, max_samples=max_samples,
                            contamination=contaminacion, random_state=s, n_jobs=-1)
        m.fit(Xs)
        sc = -m.score_samples(Xs)
        u = np.quantile(sc, 1 - contaminacion)
        conjuntos.append(set(np.where(sc >= u)[0]))
        umbrales.append(u)
        puntuaciones.append(sc)

    pares = [jaccard(conjuntos[i], conjuntos[j])
             for i in range(len(conjuntos)) for j in range(i + 1, len(conjuntos))]

    resumen = pd.DataFrame({
        "Indicador": ["Semillas evaluadas", "Registros de la muestra",
                      "Coincidencia media (Jaccard)", "Coincidencia mínima",
                      "Coincidencia máxima",
                      "Coeficiente de variación del umbral (%)"],
        "Valor": [len(semillas), f"{len(idx):,}",
                  round(np.mean(pares), 4), round(np.min(pares), 4),
                  round(np.max(pares), 4),
                  round(100 * np.std(umbrales) / np.mean(umbrales), 3)]
    })
    return resumen, pares, np.vstack(puntuaciones)


def validez_equivalente(puntuaciones, contaminacion):
    """
    Verifica si los registros que entran y salen al cambiar la semilla se
    sitúan en la frontera del umbral.

    La estabilidad imperfecta del conjunto señalado solo autoriza a hablar de
    listas de validez equivalente si los registros intercambiados son
    precisamente los que se encuentran a una distancia mínima del corte. Si
    los registros que aparecen o desaparecen estuvieran repartidos por toda
    la distribución, el resultado sería inestable en sentido fuerte.

    puntuaciones : matriz (repeticiones x registros) devuelta por
        estabilidad_semillas.
    """
    r, n = puntuaciones.shape
    umbrales = np.quantile(puntuaciones, 1 - contaminacion, axis=1)
    marcado = puntuaciones >= umbrales[:, None]
    veces = marcado.sum(axis=0)

    # Distancia relativa de cada registro al umbral, promediada entre
    # repeticiones, expresada en percentiles de la distribución.
    rangos = np.vstack([pd.Series(p).rank(pct=True).to_numpy() for p in puntuaciones])
    dist_pct = np.abs(rangos - (1 - contaminacion)).mean(axis=0) * 100

    siempre = veces == r
    nunca = veces == 0
    alguna = (veces > 0) & (veces < r)

    return pd.DataFrame({
        "Indicador": [
            "Registros señalados en todas las repeticiones (núcleo estable)",
            "Registros señalados en algunas repeticiones (zona de frontera)",
            "Registros nunca señalados",
            "Núcleo estable sobre el promedio de señalados (%)",
            "Distancia mediana al umbral del núcleo estable (percentiles)",
            "Distancia mediana al umbral de la zona de frontera (percentiles)",
            "Zona de frontera situada en el 1% más próximo al umbral (%)"],
        "Valor": [
            f"{int(siempre.sum()):,}",
            f"{int(alguna.sum()):,}",
            f"{int(nunca.sum()):,}",
            round(100 * siempre.sum() / max(marcado.sum(axis=1).mean(), 1), 2),
            round(float(np.median(dist_pct[siempre])), 4) if siempre.any() else np.nan,
            round(float(np.median(dist_pct[alguna])), 4) if alguna.any() else np.nan,
            round(100 * float((dist_pct[alguna] <= 1).mean()), 2) if alguna.any() else np.nan]
    })


def convergencia_arboles(X, contaminacion, tamanos=(25, 50, 100, 200, 400, 800),
                         max_samples=256, seed=42, n_muestra=200000, rng=None):
    """
    Evalúa la estabilidad de la puntuación al aumentar el número de árboles,
    tomando como referencia el bosque de mayor tamaño.

    El bosque de referencia debe exceder el número de árboles adoptado en la
    configuración definitiva; de lo contrario el indicador de la última fila
    es 1 por construcción y no informa sobre la convergencia.
    """
    rng = rng or np.random.default_rng(seed)
    idx = (rng.choice(len(X), min(n_muestra, len(X)), replace=False)
           if len(X) > n_muestra else np.arange(len(X)))
    Xs = X.values[idx]

    scores = {}
    for t in tamanos:
        m = IsolationForest(n_estimators=t, max_samples=max_samples,
                            contamination=contaminacion, random_state=seed, n_jobs=-1)
        m.fit(Xs)
        scores[t] = -m.score_samples(Xs)

    ref = scores[max(tamanos)]
    u_ref = np.quantile(ref, 1 - contaminacion)
    set_ref = set(np.where(ref >= u_ref)[0])

    filas = []
    for t in tamanos:
        s = scores[t]
        u = np.quantile(s, 1 - contaminacion)
        filas.append({
            "Número de árboles": t,
            "Correlación con el bosque de referencia": round(spearmanr(s, ref).statistic, 4),
            "Coincidencia de señalados (Jaccard)":
                round(jaccard(set(np.where(s >= u)[0]), set_ref), 4),
        })
    return pd.DataFrame(filas)


# ============================================================================
# 3. CONCORDANCIA ENTRE COMPONENTES
# ============================================================================
def concordancia_componentes(df, contaminacion, n_muestra=500000, rng=None):
    """Relación entre la puntuación global y la de los modelos por segmento."""
    rng = rng or np.random.default_rng(0)
    idx = (rng.choice(len(df), min(n_muestra, len(df)), replace=False)
           if len(df) > n_muestra else np.arange(len(df)))

    g = df["SCORE_GLOBAL"].values[idx]
    l = df["SCORE_LOCAL"].values[idx]

    ug = np.quantile(df["SCORE_GLOBAL"], 1 - contaminacion)
    ul = np.quantile(df["SCORE_LOCAL"], 1 - contaminacion)
    mg = df["SCORE_GLOBAL"] >= ug
    ml = df["SCORE_LOCAL"] >= ul

    return pd.DataFrame({
        "Indicador": ["Correlación de Spearman entre puntuaciones",
                      "Coincidencia de señalados (Jaccard)",
                      "Señalados solo por el modelo global",
                      "Señalados solo por los modelos locales",
                      "Señalados por ambos",
                      "Aporte exclusivo del componente local (%)"],
        "Valor": [round(spearmanr(g, l).statistic, 4),
                  round(jaccard(np.where(mg)[0], np.where(ml)[0]), 4),
                  f"{int((mg & ~ml).sum()):,}",
                  f"{int((~mg & ml).sum()):,}",
                  f"{int((mg & ml).sum()):,}",
                  round(100 * (~mg & ml).sum() / max(ml.sum(), 1), 2)]
    })


def cobertura_segmentos(df, clave, min_por_segmento):
    """Distribución del tamaño de los subespacios utilizados."""
    tam = clave.value_counts()
    con = tam[tam >= min_por_segmento]
    sin = tam[tam < min_por_segmento]
    return pd.DataFrame({
        "Indicador": ["Subespacios definidos",
                      "Con modelo local propio",
                      "Sin modelo local (por tamaño insuficiente)",
                      "Registros cubiertos por un modelo local",
                      "Cobertura de registros (%)",
                      "Tamaño mediano de los subespacios con modelo",
                      "Tamaño mínimo exigido"],
        "Valor": [f"{len(tam):,}", f"{len(con):,}", f"{len(sin):,}",
                  f"{int(con.sum()):,}",
                  round(100 * con.sum() / len(df), 2),
                  f"{int(con.median()):,}", min_por_segmento]
    })


# ============================================================================
# 4. ESTABILIDAD DE LA ATRIBUCIÓN
# ============================================================================
def estabilidad_diffi(detector, X, df_flag, variables, repeticiones=5,
                      max_registros=8000, rng=None):
    """
    Repite el cálculo de la importancia global sobre distintas submuestras y
    evalúa la coincidencia del ordenamiento resultante.
    """
    rng = rng or np.random.default_rng(0)
    matriz = []
    for _ in range(repeticiones):
        idx, idx_a, es_a = detector.muestra_diffi(df_flag, max_registros, rng)
        matriz.append(detector.diffi_global(X.loc[idx], es_a))
    M = np.vstack(matriz)

    corr = [spearmanr(M[i], M[j]).statistic
            for i in range(len(M)) for j in range(i + 1, len(M))]

    # Concentración de la importancia (índice de Herfindahl)
    media = M.mean(axis=0)
    p = media / media.sum()
    hhi = (p ** 2).sum()

    resumen = pd.DataFrame({
        "Indicador": ["Repeticiones", "Correlación media entre repeticiones",
                      "Correlación mínima", "Variable más importante",
                      "Concentración de la importancia (Herfindahl)",
                      "Variables que concentran el 80% de la importancia"],
        "Valor": [repeticiones, round(np.mean(corr), 4), round(np.min(corr), 4),
                  variables[int(media.argmax())], round(hhi, 4),
                  int((np.cumsum(np.sort(p)[::-1]) <= 0.8).sum() + 1)]
    })

    detalle = (pd.DataFrame({"Variable": variables,
                             "Importancia media": media.round(4),
                             "Desviación": M.std(axis=0).round(4)})
               .assign(**{"Coef. de variación (%)":
                          lambda d: (100 * d["Desviación"] / d["Importancia media"]).round(2)})
               .sort_values("Importancia media", ascending=False)
               .reset_index(drop=True))
    return resumen, detalle


def dispersion_atribucion(peso_dominante, umbral_multivariada=0.30):
    """Distribución del peso de la variable dominante en cada registro."""
    p = pd.Series(peso_dominante)
    return pd.DataFrame({
        "Indicador": ["Peso medio de la variable dominante",
                      "Mediana", "P25", "P75",
                      "Registros con atribución concentrada (> 50%)",
                      f"Registros con atribución dispersa (< {umbral_multivariada:.0%})"],
        "Valor": [round(p.mean(), 4), round(p.median(), 4),
                  round(p.quantile(0.25), 4), round(p.quantile(0.75), 4),
                  f"{int((p > 0.5).sum()):,}",
                  f"{int((p < umbral_multivariada).sum()):,}"]
    })


# ============================================================================
# 5. CONTRASTE FRENTE A LAS REGLAS DEL FORMATO
# ============================================================================
def incumple_reglas(df):
    """
    Máscara booleana: registros que incumplen alguna regla de validación de
    primer nivel del formato.

    Criterio alineado con el módulo calidad.py: se excluye la codificación
    convencional del campo de tecnología en los servicios sin internet fijo,
    por tratarse de una convención generalizada de la base publicada y no de
    un incumplimiento individual. Se conserva, en cambio, la ausencia de
    tecnología en los servicios que sí la requieren.
    """
    SERV_INT = [1, 4, 5, 7]
    SERV_NO_INT = [2, 3, 6]
    CAT_SEG = [101, 102, 103, 104, 105, 106, 107, 108, 109, 117]
    CAT_TEC = list(range(101, 116)) + [999]

    es_int = df["ID_SERVICIO_PAQUETE"].isin(SERV_INT)
    no_int = df["ID_SERVICIO_PAQUETE"].isin(SERV_NO_INT)
    es_117 = df["ID_SEGMENTO"] == 117

    m = (
        (~df["TRIMESTRE"].between(1, 4))
        | (~df["ID_SEGMENTO"].isin(CAT_SEG))
        | (~df["ID_SERVICIO_PAQUETE"].isin(SERV_INT + SERV_NO_INT))
        | (~df["ID_ESTADO"].isin([1, 2]))
        # Tecnología fuera del catálogo, sin contar el código cero
        | (~df["ID_TECNOLOGIA"].isin(CAT_TEC) & (df["ID_TECNOLOGIA"] != 0))
        # Servicio con internet sin tecnología declarada
        | (es_int & (df["ID_TECNOLOGIA"] == 0))
        | (es_int & (df["ID_TECNOLOGIA"] == 999))
        | (df["CANTIDAD_LINEAS_ACCESOS"] <= 0)
        | (df["VALOR_FACTURADO_O_COBRADO"] < 0)
        | (df["OTROS_VALORES_FACTURADOS"] < 0)
        | (es_int & (df["VELOCIDAD_EFECTIVA_DOWNSTREAM"] <= 0))
        | (es_int & (df["VELOCIDAD_EFECTIVA_UPSTREAM"] <= 0))
        | (no_int & (df["VELOCIDAD_EFECTIVA_DOWNSTREAM"] != 0))
        | (es_117 & ~df["ID_SERVICIO_PAQUETE"].isin([1, 2, 3]))
        | (es_117 & (df["VALOR_FACTURADO_O_COBRADO"] != 0))
    )
    return m.fillna(False)


def contraste_reglas(df, columna_anomalia="ANOMALIA"):
    """
    Contrasta los registros señalados por el modelo frente a las reglas de
    validación de primer nivel.

    La proporción de señalados que cumple todas las reglas no es, por sí
    sola, una medida del aporte del modelo: si solo una fracción reducida de
    la base incumple alguna regla, cualquier selección de registros —incluida
    una selección aleatoria— arrojaría una proporción de cumplimiento
    igualmente alta. El indicador informativo es la comparación entre el
    solapamiento observado y el que cabría esperar bajo independencia entre
    ambos mecanismos, que se calcula en contraste_observado_esperado.
    """
    incumple = incumple_reglas(df)
    marcado = df[columna_anomalia]

    n = len(df)
    a = int(marcado.sum())
    a_cumple = int((marcado & ~incumple).sum())
    a_incumple = int((marcado & incumple).sum())

    tabla = pd.DataFrame({
        "Categoría": [
            "Registros del periodo",
            "Registros que incumplen alguna regla de primer nivel",
            "Registros señalados por el modelo",
            "Señalados que incumplen alguna regla",
            "Señalados que cumplen la totalidad de las reglas",
            "Registros que incumplen y no fueron señalados"],
        "Registros": [n, int(incumple.sum()), a, a_incumple, a_cumple,
                      int((~marcado & incumple).sum())],
    })
    tabla["% del total"] = (100 * tabla["Registros"] / n).round(4)
    return tabla, incumple


def contraste_observado_esperado(df, incumple=None, columna_anomalia="ANOMALIA"):
    """
    Compara el solapamiento observado entre el conjunto señalado y el
    conjunto de incumplimientos con el esperado bajo independencia.

    Bajo independencia, el número de registros señalados que además
    incumplen alguna regla sigue una distribución hipergeométrica de
    parámetros (N, K, n), donde N es el total de registros, K el número de
    incumplimientos y n el número de señalados. La prueba establece si el
    solapamiento observado difiere del azar y en qué dirección.

    Un solapamiento significativamente inferior al esperado indica que ambos
    mecanismos operan sobre conjuntos prácticamente disjuntos, es decir, que
    el modelo identifica registros distintos de los que ya detectan las
    reglas vigentes.
    """
    if incumple is None:
        incumple = incumple_reglas(df)
    marcado = df[columna_anomalia]

    N = len(df)
    K = int(incumple.sum())
    n = int(marcado.sum())
    k = int((marcado & incumple).sum())

    esperado = n * K / N
    p_menor = float(hypergeom.cdf(k, N, K, n))
    p_mayor = float(hypergeom.sf(k - 1, N, K, n))

    def _p(v):
        """Valor p en la forma convencional de reporte."""
        if v < 0.001:
            return "< 0,001"
        return f"{v:.3f}".replace(".", ",")

    tasa_base = 100 * K / N
    tasa_senalados = 100 * k / max(n, 1)

    return pd.DataFrame({
        "Indicador": [
            "Registros del periodo (N)",
            "Registros que incumplen alguna regla (K)",
            "Registros señalados por el modelo (n)",
            "Solapamiento observado (k)",
            "Solapamiento esperado bajo independencia",
            "Razón observado / esperado",
            "Tasa de incumplimiento en la base (%)",
            "Tasa de incumplimiento entre los señalados (%)",
            "Valor p (solapamiento inferior al esperado)",
            "Valor p (solapamiento superior al esperado)"],
        "Valor": [
            f"{N:,}", f"{K:,}", f"{n:,}", f"{k:,}",
            round(esperado, 1),
            round(k / max(esperado, 1e-9), 3),
            round(tasa_base, 3),
            round(tasa_senalados, 3),
            _p(p_menor), _p(p_mayor)]
    })


def contraste_por_tipologia(anom, incumple_serie):
    """Distribución del cumplimiento de las reglas según el tipo de anomalía."""
    a = anom.copy()
    a["Cumple las reglas"] = ~incumple_serie.reindex(a.index).fillna(False)
    t = (pd.crosstab(a["TIPOLOGIA"], a["Cumple las reglas"])
           .rename(columns={True: "Cumple todas las reglas",
                            False: "Incumple alguna regla"})
           .reset_index().rename(columns={"TIPOLOGIA": "Tipo de anomalía"}))
    if "Cumple todas las reglas" in t and "Incumple alguna regla" in t:
        t["% que cumple"] = (100 * t["Cumple todas las reglas"]
                             / (t["Cumple todas las reglas"]
                                + t["Incumple alguna regla"])).round(2)
    return t
