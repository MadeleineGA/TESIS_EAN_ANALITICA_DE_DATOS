"""
validacion.py — Validación del modelo mediante inyección controlada de
anomalías y comparación con algoritmos alternativos
Reporte T.1.3 de servicios fijos · CRC

El reporte no dispone de un conjunto de registros previamente clasificados
como correctos o erróneos, de modo que las medidas de desempeño supervisado
no son calculables sobre los datos tal como fueron entregados. Este módulo
construye esa referencia de forma controlada: parte de una base depurada,
introduce en ella un número conocido de registros alterados conforme a los
errores de diligenciamiento documentados en el diagnóstico, y evalúa qué
proporción de ellos recupera el modelo.

El procedimiento no sustituye la verificación por juicio experto. Su alcance
es acotado: establece la capacidad del modelo para recuperar errores del
tipo que se le introduce, bajo el supuesto de que la base de partida está
libre de errores de esa misma naturaleza.

Escenarios de inyección:
    unidad        confusión en la unidad de medida de la velocidad
    intercambio   permuta entre velocidad de descarga y de carga
    facturacion   valor facturado desproporcionado frente a los accesos
    accesos       error de magnitud en el número de accesos
    contextual    velocidad plausible en el agregado nacional pero muy
                  superior al promedio de su municipio
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, average_precision_score

from modelo import construir_variables
from metricas import incumple_reglas

ESCENARIOS = ["unidad", "intercambio", "facturacion", "accesos", "contextual"]

NOMBRES_ESCENARIO = {
    "unidad": "Confusión en la unidad de medida de la velocidad",
    "intercambio": "Cambio entre la velocidad de descarga y la de carga",
    "facturacion": "Valor facturado desproporcionado frente a los accesos",
    "accesos": "Error de magnitud en el número de accesos",
    "contextual": "Velocidad muy superior al promedio de su municipio",
}

MAGNITUDES_ESCENARIO = {"unidad": 1000, "intercambio": 1,
                        "facturacion": 100, "accesos": 1000,
                        "contextual": 10}

# ============================================================================
# BASE DE PARTIDA
# ============================================================================
def base_limpia(df, n=400000, seed=42):
    """
    Submuestra de registros que satisfacen la totalidad de las reglas de
    primer nivel y presentan velocidad y accesos positivos.

    La base resultante no está libre de anomalías —precisamente el trabajo
    sostiene que las reglas no las detectan—, de modo que las medidas
    obtenidas sobre ella constituyen una cota inferior del desempeño: un
    registro no inyectado que el modelo señale se cuenta como falso
    positivo, aunque pueda corresponder a una anomalía real.
    """
    rng = np.random.default_rng(seed)
    ok = ~incumple_reglas(df)
    base = df[ok & df["ES_INTERNET"]
              & (df["VELOCIDAD_EFECTIVA_DOWNSTREAM"] > 0)
              & (df["CANTIDAD_LINEAS_ACCESOS"] > 0)].copy()
    if len(base) > n:
        base = base.iloc[rng.choice(len(base), n, replace=False)]
    base = base.reset_index(drop=True)
    # Los campos que la inyección modifica se llevan a doble precisión: la
    # lectura original los define en precisión simple y la alteración produce
    # magnitudes que no caben en esa escala.
    for c in ["VELOCIDAD_EFECTIVA_DOWNSTREAM", "VELOCIDAD_EFECTIVA_UPSTREAM",
              "CANTIDAD_LINEAS_ACCESOS", "VALOR_FACTURADO_O_COBRADO",
              "OTROS_VALORES_FACTURADOS"]:
        if c in base.columns:
            base[c] = base[c].astype("float64")
    return base


def _elegir(n_total, n, rng, condicion=None):
    """Posiciones de las filas que se alteran."""
    pos = (np.arange(n_total) if condicion is None
           else np.flatnonzero(np.asarray(condicion)))
    return rng.choice(pos, min(n, len(pos)), replace=False)


# ============================================================================
# ESCENARIOS DE INYECCIÓN
# ============================================================================
def inyectar(base, escenario, n=2000, magnitud=1000, promedios=None, seed=42):
    """
    Devuelve (base_modificada, etiqueta) donde etiqueta vale True en los
    registros alterados.

    magnitud : factor de la alteración. Su valor se interpreta según el
        escenario; en 'intercambio' no interviene.
    """
    rng = np.random.default_rng(seed)
    d = base.copy().reset_index(drop=True)
    y = np.zeros(len(d), dtype=bool)
    N = len(d)

    cdown = d.columns.get_loc("VELOCIDAD_EFECTIVA_DOWNSTREAM")
    cup = d.columns.get_loc("VELOCIDAD_EFECTIVA_UPSTREAM")
    cacc = d.columns.get_loc("CANTIDAD_LINEAS_ACCESOS")
    cval = d.columns.get_loc("VALOR_FACTURADO_O_COBRADO")

    if escenario == "unidad":
        sel = _elegir(N, n, rng)
        d.iloc[sel, cdown] = d.iloc[sel, cdown].to_numpy() * magnitud
        d.iloc[sel, cup] = d.iloc[sel, cup].to_numpy() * magnitud

    elif escenario == "intercambio":
        cond = (d["VELOCIDAD_EFECTIVA_DOWNSTREAM"].to_numpy()
                > 2 * d["VELOCIDAD_EFECTIVA_UPSTREAM"].to_numpy())
        sel = _elegir(N, n, rng, cond)
        down = d.iloc[sel, cdown].to_numpy().copy()
        up = d.iloc[sel, cup].to_numpy().copy()
        d.iloc[sel, cdown] = up
        d.iloc[sel, cup] = down

    elif escenario == "facturacion":
        cond = d["VALOR_FACTURADO_O_COBRADO"].to_numpy() > 0
        sel = _elegir(N, n, rng, cond)
        d.iloc[sel, cval] = d.iloc[sel, cval].to_numpy() * magnitud

    elif escenario == "accesos":
        sel = _elegir(N, n, rng)
        d.iloc[sel, cacc] = np.round(d.iloc[sel, cacc].to_numpy() * magnitud)

    elif escenario == "contextual":
        if promedios is None:
            raise ValueError("El escenario contextual requiere los promedios "
                             "municipales.")
        claves = ["ANNO", "TRIMESTRE", "ID_MUNICIPIO", "SEG_GRUPO"]
        aux = d[claves].merge(promedios[claves + ["VEL_PROMEDIO"]],
                              on=claves, how="left")
        # Se seleccionan registros de municipios con promedio bajo y se les
        # asigna una velocidad que resulta corriente en el agregado nacional
        # pero muy superior a la de su municipio.
        prm = aux["VEL_PROMEDIO"].to_numpy(dtype="float64")
        corte = np.nanpercentile(prm, 40)
        cond = (prm > 0) & (prm < corte)
        sel = _elegir(N, n, rng, cond)
        d.iloc[sel, cdown] = np.round(prm[sel] * magnitud, 2)

    else:
        raise ValueError(f"Escenario no reconocido: {escenario}")

    y[sel] = True
    return d, y


# ============================================================================
# ALGORITMOS EVALUADOS
# ============================================================================
def _estandarizar(X):
    Xv = X.values.astype("float64") if hasattr(X, "values") else X.astype("float64")
    mu = Xv.mean(axis=0)
    sd = Xv.std(axis=0)
    sd[sd == 0] = 1.0
    return (Xv - mu) / sd


def puntuar_isolation_forest(X, cfg_arboles=400, max_samples=256, seed=42,
                             contaminacion=0.01):
    m = IsolationForest(n_estimators=cfg_arboles, max_samples=max_samples,
                        contamination=contaminacion, random_state=seed, n_jobs=-1)
    m.fit(X.values)
    return -m.score_samples(X.values)


def puntuar_hbos(X, n_bins=20):
    """
    Histogram-Based Outlier Score (Goldstein y Dengel).

    Estima la densidad de cada variable por separado mediante un histograma
    de anchura variable definido por cuantiles —variante adecuada ante
    distribuciones muy asimétricas— y agrega los logaritmos de las
    densidades inversas. Supone independencia entre variables, por lo que no
    detecta anomalías definidas por la relación entre campos.
    """
    Xv = X.values.astype("float64") if hasattr(X, "values") else X.astype("float64")
    score = np.zeros(len(Xv))
    for j in range(Xv.shape[1]):
        v = Xv[:, j]
        bordes = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
        if len(bordes) < 3:
            continue
        conteo, bordes = np.histogram(v, bins=bordes)
        ancho = np.diff(bordes)
        dens = conteo / (len(v) * np.maximum(ancho, 1e-12))
        dens = dens / max(dens.max(), 1e-12)
        pos = np.clip(np.digitize(v, bordes[1:-1]), 0, len(dens) - 1)
        score += np.log(1.0 / np.maximum(dens[pos], 1e-12))
    return score


def puntuar_lof(X, n_vecinos=20, n_muestra=60000, seed=42):
    """
    Local Outlier Factor sobre una submuestra.

    El cálculo exige las distancias entre vecinos, cuyo costo crece de forma
    cuadrática con el número de registros; por ello se evalúa sobre una
    submuestra y se ajusta en modo de novedad para puntuar el resto.
    """
    rng = np.random.default_rng(seed)
    Xv = _estandarizar(X)
    idx = rng.choice(len(Xv), min(n_muestra, len(Xv)), replace=False)
    m = LocalOutlierFactor(n_neighbors=n_vecinos, novelty=True, n_jobs=-1)
    m.fit(Xv[idx])
    return -m.score_samples(Xv)


def puntuar_pca(X, n_componentes=None, seed=42):
    """
    Error de reconstrucción de un análisis de componentes principales.

    Mide la distancia de cada registro al subespacio que explica la mayor
    parte de la varianza del conjunto. Es el enfoque de referencia para
    anomalías definidas como desviaciones de la estructura lineal de
    correlación.
    """
    Xv = _estandarizar(X)
    k = n_componentes or max(1, Xv.shape[1] // 2)
    p = PCA(n_components=k, random_state=seed)
    rec = p.inverse_transform(p.fit_transform(Xv))
    return ((Xv - rec) ** 2).sum(axis=1)


ALGORITMOS = {
    "Isolation Forest": puntuar_isolation_forest,
    "HBOS": puntuar_hbos,
    "LOF (submuestra)": puntuar_lof,
    "PCA (error de reconstrucción)": puntuar_pca,
}


# ============================================================================
# MEDIDAS DE DESEMPEÑO
# ============================================================================
def medidas(score, y, contaminacion=0.01):
    """ROC-AUC, precisión promedio y desempeño en el umbral de operación."""
    u = np.quantile(score, 1 - contaminacion)
    marc = score >= u
    vp = int((marc & y).sum())
    return {
        "ROC-AUC": round(float(roc_auc_score(y, score)), 4),
        "Precisión promedio": round(float(average_precision_score(y, score)), 4),
        "Exhaustividad en el umbral del 1% (%)": round(100 * vp / max(y.sum(), 1), 2),
        "Precisión en el umbral del 1% (%)": round(100 * vp / max(marc.sum(), 1), 2),
        "Señalados en el umbral": int(marc.sum()),
        "Inyectados recuperados": vp,
    }


# ============================================================================
# EXPERIMENTOS
# ============================================================================
def evaluar_escenarios(base, prom, cfg, escenarios=None, magnitudes=None):
    """
    Desempeño del Isolation Forest en cada escenario de inyección.

    magnitudes : diccionario {escenario: magnitud}. Si es None se emplean
        magnitudes compatibles con los casos observados en el diagnóstico.
    """
    escenarios = escenarios or ESCENARIOS
    magnitudes = magnitudes or MAGNITUDES_ESCENARIO
    filas = []
    for e in escenarios:
        d, y = inyectar(base, e, n=cfg.N_INYECCION,
                        magnitud=magnitudes[e], promedios=prom, seed=cfg.SEED)
        X, _ = construir_variables(d, promedios_municipales=prom, cfg=cfg)
        sc = puntuar_isolation_forest(X, cfg.N_ARBOLES, cfg.MAX_SAMPLES,
                                      cfg.SEED, cfg.CONTAMINACION)
        m = medidas(sc, y, cfg.CONTAMINACION)
        filas.append({"Escenario": NOMBRES_ESCENARIO[e],
                      "Inyectados": int(y.sum()),
                      "Magnitud": magnitudes[e], **m})
    return pd.DataFrame(filas)


def curva_magnitud(base, prom, cfg, escenario="unidad", magnitudes=None):
    """
    Desempeño en función de la magnitud de la alteración introducida.

    Permite establecer a partir de qué tamaño de error el modelo lo recupera
    y, por tanto, qué desviaciones quedan por debajo de su capacidad de
    detección.
    """
    magnitudes = magnitudes or cfg.MAGNITUDES
    filas = []
    for g in magnitudes:
        d, y = inyectar(base, escenario, n=cfg.N_INYECCION, magnitud=g,
                        promedios=prom, seed=cfg.SEED)
        X, _ = construir_variables(d, promedios_municipales=prom, cfg=cfg)
        sc = puntuar_isolation_forest(X, cfg.N_ARBOLES, cfg.MAX_SAMPLES,
                                      cfg.SEED, cfg.CONTAMINACION)
        m = medidas(sc, y, cfg.CONTAMINACION)
        filas.append({"Factor de alteración": g, **m})
    return pd.DataFrame(filas)


def comparar_algoritmos(base, prom, cfg, escenarios=None, magnitudes=None):
    """
    Compara el Isolation Forest con algoritmos alternativos sobre los mismos
    conjuntos inyectados.

    Sustenta empíricamente la selección del algoritmo: los criterios
    expuestos en el marco teórico se establecen sobre las propiedades
    documentadas de cada método, mientras que esta comparación los contrasta
    con su comportamiento en los datos del reporte.
    """
    escenarios = escenarios or ESCENARIOS
    magnitudes = magnitudes or {"unidad": 1000, "intercambio": 1,
                                "facturacion": 100, "accesos": 1000,
                                "contextual": 10}
    filas = []
    for e in escenarios:
        d, y = inyectar(base, e, n=cfg.N_INYECCION, magnitud=magnitudes[e],
                        promedios=prom, seed=cfg.SEED)
        X, _ = construir_variables(d, promedios_municipales=prom, cfg=cfg)
        for nombre, fn in ALGORITMOS.items():
            if nombre == "Isolation Forest":
                sc = fn(X, cfg.N_ARBOLES, cfg.MAX_SAMPLES, cfg.SEED,
                        cfg.CONTAMINACION)
            else:
                sc = fn(X)
            m = medidas(sc, y, cfg.CONTAMINACION)
            filas.append({"Escenario": NOMBRES_ESCENARIO[e],
                          "Algoritmo": nombre,
                          "ROC-AUC": m["ROC-AUC"],
                          "Precisión promedio": m["Precisión promedio"],
                          "Exhaustividad en el umbral del 1% (%)":
                              m["Exhaustividad en el umbral del 1% (%)"]})
    t = pd.DataFrame(filas)
    resumen = (t.groupby("Algoritmo", as_index=False)
                 .agg(**{"ROC-AUC medio": ("ROC-AUC", "mean"),
                         "ROC-AUC mínimo": ("ROC-AUC", "min"),
                         "Precisión promedio media": ("Precisión promedio", "mean"),
                         "Exhaustividad media (%)":
                             ("Exhaustividad en el umbral del 1% (%)", "mean")})
                 .round(4).sort_values("ROC-AUC medio", ascending=False))
    return t, resumen


def contraste_segmentacion(base, prom, cfg, escenario="contextual",
                           magnitud=10):
    """
    Desempeño del componente global frente al segmentado por municipio sobre
    anomalías de carácter contextual.

    Contrasta empíricamente la limitación del Isolation Forest ante
    anomalías locales señalada en el marco teórico: las anomalías inyectadas
    son plausibles en el agregado nacional y solo resultan atípicas dentro de
    su municipio.
    """
    from modelo import DetectorAnomalias

    d, y = inyectar(base, escenario, n=cfg.N_INYECCION, magnitud=magnitud,
                    promedios=prom, seed=cfg.SEED)
    X, _ = construir_variables(d, promedios_municipales=prom, cfg=cfg)

    det = DetectorAnomalias(cfg.N_ARBOLES, cfg.MAX_SAMPLES, cfg.CONTAMINACION,
                            cfg.SEED)
    sg = det.ajustar_global(X)
    clave = d[cfg.SEGMENTO_LOCAL].astype(str).agg("|".join, axis=1)
    sl, n_mod, cob = det.ajustar_local(X, clave, cfg.MIN_POR_SEGMENTO,
                                       verbose=False)
    sl = sl.fillna(pd.Series(sg, index=X.index))

    filas = []
    for nombre, sc in [("Solo global", pd.Series(sg, index=X.index)),
                       ("Solo segmentado por municipio", sl),
                       (f"Combinado {int((1-cfg.PESO_LOCAL)*100)}/"
                        f"{int(cfg.PESO_LOCAL*100)}",
                        det.combinar(pd.Series(sg, index=X.index), sl,
                                     cfg.PESO_LOCAL))]:
        m = medidas(sc.to_numpy(), y, cfg.CONTAMINACION)
        filas.append({"Configuración": nombre, **m})
    return pd.DataFrame(filas), n_mod, cob
