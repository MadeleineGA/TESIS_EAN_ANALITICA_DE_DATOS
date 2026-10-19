"""
modelo.py — Modelo de detección de anomalías
Reporte T.1.3 de servicios fijos · CRC

Contiene:
    construir_variables   Ingeniería de variables de entrada
    DetectorAnomalias     Isolation Forest global y segmentado (Layered IF)
                          Atribución por variable (DIFFI)
    clasificar_tipologia  Asignación del tipo de anomalía
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Topes por defecto de las variables derivadas; pueden sobreescribirse
# entregando el módulo de configuración a construir_variables.
TOPE_VALOR_POR_ACCESO = 1e7
TOPE_RATIO_DOWN_UP = 1000
TOPE_VEL_REL_MUNICIPIO = 100
TOPE_VEL_REL_TECNOLOGIA = 100


# ============================================================================
# INGENIERÍA DE VARIABLES
# ============================================================================
def construir_variables(df, promedios_municipales=None, cfg=None,
                        log_velocidades=None):
    """
    Construye la matriz de entrada del modelo a partir del reporte.

    df : DataFrame con las columnas del reporte, más SEG_GRUPO y ES_INTERNET.
    promedios_municipales : DataFrame con ANNO, TRIMESTRE, ID_MUNICIPIO,
        SEG_GRUPO y VEL_PROMEDIO. Si se entrega, se incorpora la velocidad
        relativa al promedio de su municipio, segmento y trimestre.
    cfg : módulo de configuración del que se toman los topes y la opción de
        transformación de las velocidades. Si es None se emplean los valores
        por defecto definidos en este módulo.
    log_velocidades : si es True, las velocidades absolutas entran como
        log(1+v); si es False, en su escala original. Permite documentar el
        efecto de esta decisión de preprocesamiento sobre la atribución por
        variable. Si es None se toma de cfg.LOG_VELOCIDADES.

    Devuelve (X, nombres_de_variables).
    """
    tope_vpa = getattr(cfg, "TOPE_VALOR_POR_ACCESO", TOPE_VALOR_POR_ACCESO)
    tope_ratio = getattr(cfg, "TOPE_RATIO_DOWN_UP", TOPE_RATIO_DOWN_UP)
    tope_mun = getattr(cfg, "TOPE_VEL_REL_MUNICIPIO", TOPE_VEL_REL_MUNICIPIO)
    tope_tec = getattr(cfg, "TOPE_VEL_REL_TECNOLOGIA", TOPE_VEL_REL_TECNOLOGIA)
    if log_velocidades is None:
        log_velocidades = getattr(cfg, "LOG_VELOCIDADES", False)

    X = pd.DataFrame(index=df.index)

    # --- Velocidades absolutas ---
    # Su tratamiento lo define cfg.LOG_VELOCIDADES. Por defecto se conservan
    # en su escala original: el formato no las acota por arriba y los valores
    # extremos observados —que superan el millón de Mbps— constituyen el
    # objeto de detección, de modo que comprimirlos atenuaría la señal.
    for c in ["VELOCIDAD_EFECTIVA_DOWNSTREAM", "VELOCIDAD_EFECTIVA_UPSTREAM"]:
        if c in df.columns:
            v = df[c].fillna(0).clip(lower=0)
            if log_velocidades:
                X["LOG_" + c] = np.log1p(v)
            else:
                X[c] = v

    # --- Transformación logarítmica de variables con fuerte asimetría ---
    for c in ["CANTIDAD_LINEAS_ACCESOS", "VALOR_FACTURADO_O_COBRADO",
              "OTROS_VALORES_FACTURADOS"]:
        if c in df.columns:
            X["LOG_" + c] = np.log1p(df[c].fillna(0).clip(lower=0))

    # --- Razones derivadas ---
    if {"VALOR_FACTURADO_O_COBRADO", "CANTIDAD_LINEAS_ACCESOS"} <= set(df.columns):
        v = (df["VALOR_FACTURADO_O_COBRADO"]
             / df["CANTIDAD_LINEAS_ACCESOS"].replace(0, np.nan))
        X["VALOR_POR_ACCESO"] = (v.replace([np.inf, -np.inf], np.nan)
                                  .fillna(0).clip(0, tope_vpa))

    if {"VELOCIDAD_EFECTIVA_DOWNSTREAM", "VELOCIDAD_EFECTIVA_UPSTREAM"} <= set(df.columns):
        r = (df["VELOCIDAD_EFECTIVA_DOWNSTREAM"]
             / df["VELOCIDAD_EFECTIVA_UPSTREAM"].replace(0, np.nan))
        X["RATIO_DOWN_UP"] = (r.replace([np.inf, -np.inf], np.nan)
                               .fillna(0).clip(0, tope_ratio))

    # --- Velocidad relativa al promedio de su municipio, segmento y trimestre ---
    if promedios_municipales is not None:
        claves = ["ANNO", "TRIMESTRE", "ID_MUNICIPIO", "SEG_GRUPO"]
        aux = df[claves].merge(
            promedios_municipales[claves + ["VEL_PROMEDIO"]],
            on=claves, how="left")
        rel = (df["VELOCIDAD_EFECTIVA_DOWNSTREAM"].values
               / aux["VEL_PROMEDIO"].replace(0, np.nan).values)
        X["VEL_RELATIVA_MUNICIPIO"] = (pd.Series(rel, index=df.index)
                                       .replace([np.inf, -np.inf], np.nan)
                                       .fillna(0).clip(0, tope_mun))

    # --- Velocidad relativa a la mediana de su tecnología ---
    if {"ID_TECNOLOGIA", "VELOCIDAD_EFECTIVA_DOWNSTREAM", "ES_INTERNET"} <= set(df.columns):
        med_tec = (df[df["ES_INTERNET"]]
                   .groupby("ID_TECNOLOGIA")["VELOCIDAD_EFECTIVA_DOWNSTREAM"]
                   .median())
        med_tec = med_tec.replace(0, np.nan)
        X["VEL_RELATIVA_TECNOLOGIA"] = (
            df["VELOCIDAD_EFECTIVA_DOWNSTREAM"] / df["ID_TECNOLOGIA"].map(med_tec)
        ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, tope_tec)

    # --- Codificación por frecuencia de las variables categóricas ---
    # Se excluye ID_TECNOLOGIA: la codificación por frecuencia introduce un
    # sesgo hacia las tecnologías minoritarias, que resultan señaladas por su
    # rareza y no por el contenido del registro. Su información se incorpora
    # mediante VEL_RELATIVA_TECNOLOGIA.
    for c in ["ID_SEGMENTO", "ID_SERVICIO_PAQUETE", "ID_ESTADO"]:
        if c in df.columns:
            frec = df[c].value_counts(normalize=True)
            X["FREQ_" + c] = df[c].map(frec).fillna(0)

    X = X.astype("float32")
    return X, list(X.columns)


# ============================================================================
# DETECTOR
# ============================================================================
class DetectorAnomalias:
    """
    Isolation Forest global y segmentado, con atribución por variable.

    El componente segmentado implementa una aproximación al Layered Isolation
    Forest (T. Liu et al., 2024): en lugar de dividir geométricamente el
    espacio, los subespacios se definen por una variable del reporte.
    """

    def __init__(self, n_arboles=400, max_samples=256, contaminacion=0.01,
                 seed=42, n_arboles_local=100):
        self.n_arboles = n_arboles
        self.max_samples = max_samples
        self.contaminacion = contaminacion
        self.seed = seed
        self.n_arboles_local = n_arboles_local
        self.modelo_global = None
        self.variables = None

    # ------------------------------------------------------------- GLOBAL
    def ajustar_global(self, X):
        """Entrena el bosque global y devuelve la puntuación de cada registro."""
        self.variables = list(X.columns)
        self.modelo_global = IsolationForest(
            n_estimators=self.n_arboles, max_samples=self.max_samples,
            contamination=self.contaminacion, random_state=self.seed, n_jobs=-1)
        self.modelo_global.fit(X.values)
        return -self.modelo_global.score_samples(X.values)

    # -------------------------------------------------------------- LOCAL
    def ajustar_local(self, X, clave, min_por_segmento=200, verbose=True):
        """
        Entrena un bosque independiente por subespacio.

        X     : matriz de entrada
        clave : Serie con el identificador de subespacio de cada registro
        Devuelve (scores, n_modelos, cobertura_pct).
        """
        scores = pd.Series(np.nan, index=X.index)
        tam = clave.value_counts()
        validos = tam[tam >= min_por_segmento].index

        for i, s in enumerate(validos, 1):
            idx = clave[clave == s].index
            m = IsolationForest(
                n_estimators=self.n_arboles_local,
                max_samples=min(self.max_samples, len(idx)),
                contamination=self.contaminacion, random_state=self.seed, n_jobs=-1)
            Xi = X.loc[idx].values
            m.fit(Xi)
            scores.loc[idx] = -m.score_samples(Xi)
            if verbose and i % 200 == 0:
                print(f"    modelos locales: {i:,}/{len(validos):,}")

        cobertura = scores.notna().mean() * 100
        return scores, len(validos), cobertura

    # ---------------------------------------------------------- COMBINAR
    @staticmethod
    def combinar(score_global, score_local, peso_local=0.5):
        """
        Combina ambas puntuaciones tras normalizarlas a un rango común.

        La normalización es necesaria porque ambas puntuaciones se producen
        sobre conjuntos de entrenamiento distintos y no son directamente
        comparables en su escala original.
        """
        def norm(s):
            return (s - s.min()) / (s.max() - s.min() + 1e-12)
        return (1 - peso_local) * norm(score_global) + peso_local * norm(score_local)

    # ------------------------------------------------------------- DIFFI
    @staticmethod
    def _profundidades(arbol):
        d = np.zeros(arbol.node_count, dtype=np.int32)
        pila = [(0, 0)]
        while pila:
            n, p = pila.pop()
            d[n] = p
            if arbol.children_left[n] != -1:
                pila.append((arbol.children_left[n], p + 1))
                pila.append((arbol.children_right[n], p + 1))
        return d

    def diffi_global(self, X, es_anomalia):
        """
        Importancia global de cada variable (Carletti et al., 2023).

        Compara el uso ponderado de cada variable en los caminos de los
        registros anómalos frente al de los registros normales. Valores
        superiores a 1 indican mayor contribución al aislamiento de anomalías.
        """
        Xv = X.values if hasattr(X, "values") else X
        nf = Xv.shape[1]
        imp_a = np.zeros(nf); cnt_a = np.zeros(nf)
        imp_n = np.zeros(nf); cnt_n = np.zeros(nf)

        for est in self.modelo_global.estimators_:
            t = est.tree_
            prof = self._profundidades(t)
            ind = est.decision_path(Xv)
            filas = np.repeat(np.arange(Xv.shape[0]), np.diff(ind.indptr))
            nodos = ind.indices
            feats = t.feature[nodos]
            val = feats >= 0
            w = 1.0 / (prof[nodos[val]] + 1.0)
            f = feats[val]
            ea = es_anomalia[filas[val]]
            imp_a += np.bincount(f[ea], weights=w[ea], minlength=nf)
            cnt_a += np.bincount(f[ea], minlength=nf)
            imp_n += np.bincount(f[~ea], weights=w[~ea], minlength=nf)
            cnt_n += np.bincount(f[~ea], minlength=nf)

        ratio_a = imp_a / np.maximum(cnt_a, 1)
        ratio_n = imp_n / np.maximum(cnt_n, 1)
        return ratio_a / np.maximum(ratio_n, 1e-12)

    def diffi_local(self, X):
        """
        Contribución de cada variable a cada registro, normalizada a 1.

        Cada uso de variable se pondera por la reducción del tamaño del nodo
        que produce la partición: los cortes que aíslan la observación de la
        mayor parte de los datos reciben mayor peso. Sin esta ponderación la
        atribución resulta prácticamente uniforme entre variables, porque
        todos los registros recorren un número similar de nodos.
        """
        Xv = X.values if hasattr(X, "values") else X
        nf = Xv.shape[1]
        acum = np.zeros((Xv.shape[0], nf))

        for est in self.modelo_global.estimators_:
            t = est.tree_
            n_nodo = t.n_node_samples
            ind = est.decision_path(Xv)
            filas = np.repeat(np.arange(Xv.shape[0]), np.diff(ind.indptr))
            nodos = ind.indices
            feats = t.feature[nodos]
            val = feats >= 0

            # Nodo hijo por el que continúa cada observación
            n_pad = n_nodo[nodos[val]]
            izq = t.children_left[nodos[val]]
            der = t.children_right[nodos[val]]
            umbral = t.threshold[nodos[val]]
            va_izq = Xv[filas[val], feats[val]] <= umbral
            n_hij = np.where(va_izq, n_nodo[izq], n_nodo[der])

            # Peso: proporción de datos eliminados por la partición
            w = (n_pad - n_hij) / np.maximum(n_pad, 1)
            np.add.at(acum, (filas[val], feats[val]), w)

        return acum / np.maximum(acum.sum(axis=1, keepdims=True), 1e-12)

    def diffi_local_bloques(self, X, indices, tam_bloque=5000):
        """
        Atribución local sobre un conjunto de índices, calculada por bloques.

        Evita construir en memoria la matriz de caminos de todos los
        registros a la vez. Devuelve la matriz de contribuciones en el orden
        de `indices`.
        """
        bloques = [self.diffi_local(X.loc[indices[i:i + tam_bloque]])
                   for i in range(0, len(indices), tam_bloque)]
        return np.vstack(bloques)

    def muestra_diffi(self, df_flag, max_registros=8000, rng=None):
        """Selecciona una muestra equilibrada de anómalos y normales."""
        rng = rng or np.random.default_rng(self.seed)
        idx_a = df_flag.index[df_flag].to_numpy()
        idx_n = df_flag.index[~df_flag].to_numpy()
        if len(idx_a) > max_registros // 2:
            idx_a = rng.choice(idx_a, max_registros // 2, replace=False)
        idx_n = rng.choice(idx_n, min(len(idx_n), max_registros // 2), replace=False)
        idx = np.concatenate([idx_a, idx_n])
        return idx, idx_a, np.isin(idx, idx_a)


# ============================================================================
# TIPOLOGÍA
# ============================================================================
def clasificar_tipologia(df, anom, umbral_global, cfg):
    """
    Asigna a cada anomalía un tipo según la tipología del marco teórico.

    El orden jerárquico efectivo es: temporal, contextual, multivariada,
    colectiva y, en ausencia de las anteriores, puntual. Las categorías no
    son mutuamente excluyentes por definición —un registro puede cumplir
    varias condiciones— y la jerarquía las vuelve excluyentes por
    construcción, de modo que cada registro recibe un único tipo.

    La categoría colectiva se define como tasa de señalamiento del grupo
    proveedor–trimestre y no como conteo absoluto: un criterio de conteo
    señalaría como colectivos a los proveedores de mayor tamaño por el solo
    volumen de registros que reportan.
    """
    # --- Temporal: variación abrupta frente al trimestre anterior ---
    g = [c for c in ["ID_EMPRESA", "ID_MUNICIPIO", "ID_SERVICIO_PAQUETE"]
         if c in df.columns]
    serie = (df.groupby(g + ["ORDEN"], observed=True)["CANTIDAD_LINEAS_ACCESOS"]
               .sum().reset_index().sort_values(g + ["ORDEN"]))
    serie["PREVIO"] = serie.groupby(g, observed=True)["CANTIDAD_LINEAS_ACCESOS"].shift(1)
    serie["VAR_PCT"] = ((serie["CANTIDAD_LINEAS_ACCESOS"] - serie["PREVIO"])
                        / serie["PREVIO"].replace(0, np.nan) * 100)
    saltos = serie[(serie["PREVIO"] >= cfg.BASE_TEMPORAL)
                   & (serie["VAR_PCT"].abs() > cfg.UMBRAL_TEMPORAL)]
    clave_salto = set(map(tuple, saltos[g + ["ORDEN"]].values))
    anom["ES_TEMPORAL"] = [tuple(r) in clave_salto for r in anom[g + ["ORDEN"]].values]

    # --- Colectiva: tasa de señalamiento del proveedor en el trimestre ---
    senal = anom.groupby(["ID_EMPRESA", "ORDEN"], observed=True).size()
    total = df.groupby(["ID_EMPRESA", "ORDEN"], observed=True).size()
    tasa = (senal / total.reindex(senal.index)).rename("TASA")
    tasa_general = len(anom) / len(df)
    umbral_tasa = cfg.FACTOR_COLECTIVA * tasa_general

    grupos = set(tasa[(tasa >= umbral_tasa)
                      & (senal >= cfg.MIN_COLECTIVA)].index)
    anom["ES_COLECTIVA"] = [(e, o) in grupos
                            for e, o in zip(anom["ID_EMPRESA"], anom["ORDEN"])]

    # --- Contextual: alta puntuación local pero baja global ---
    anom["ES_CONTEXTUAL"] = anom["SCORE_GLOBAL"] < umbral_global

    # --- Multivariada: contribución repartida entre varias variables ---
    anom["ES_MULTIVARIADA"] = anom["PESO_DOMINANTE"] < cfg.UMBRAL_MULTIVAR

    def tipificar(r):
        if r["ES_TEMPORAL"]:
            return "Temporal"
        if r["ES_CONTEXTUAL"]:
            return "Contextual"
        if r["ES_MULTIVARIADA"]:
            return "Multivariada"
        if r["ES_COLECTIVA"]:
            return "Colectiva"
        return "Puntual"

    anom["TIPOLOGIA"] = anom.apply(tipificar, axis=1)

    detalle_colectiva = (pd.DataFrame({"Señalados": senal, "Reportados": total.reindex(senal.index),
                                       "Tasa (%)": (100 * tasa).round(2)})
                         .reset_index()
                         .assign(**{"Supera el criterio":
                                    lambda d: np.where(
                                        (d["Tasa (%)"] >= 100 * umbral_tasa)
                                        & (d["Señalados"] >= cfg.MIN_COLECTIVA),
                                        "Sí", "No")})
                         .sort_values("Señalados", ascending=False))

    return anom, saltos, detalle_colectiva, umbral_tasa
