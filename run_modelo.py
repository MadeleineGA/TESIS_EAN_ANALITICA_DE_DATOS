"""
run_modelo.py — Informe del modelo de detección de anomalías del reporte de
servicios fijos

Genera: salidas/informe_modelo.html y salidas/anomalias_detectadas.csv

Entrena el Isolation Forest global y los modelos por municipio, calcula la
atribución por variable (DIFFI), clasifica las anomalías según la tipología
y produce las métricas de evaluación.

Requiere haber ejecutado antes run_calidad.py, que deja preparados los
archivos de salidas/. Si no existen, el script vuelve a leer el CSV original.

La numeración de tablas y figuras es automática y consecutiva.

Uso:
    python run_modelo.py
"""

import numpy as np
import pandas as pd

import config as cfg
import informe as inf
import metricas
import comparacion
from modelo import DetectorAnomalias, construir_variables, clasificar_tipologia

rng = np.random.default_rng(cfg.SEED)
rep = inf.Informe(
    "Modelo de detección de anomalías en el reporte de servicios fijos",
    subtitulo="Reporte T.1.3 · Comisión de Regulación de Comunicaciones · "
              f"periodo {cfg.ANNO_INI}–{cfg.ANNO_FIN}")

# ============================================================================
# 0. CARGA
# ============================================================================
df, prom = inf.cargar_o_preparar(cfg)

# --- Delimitación al segmento residencial ---
if cfg.SEGMENTO_MODELO:
    n_antes = len(df)
    df = df[df["SEG_GRUPO"] == cfg.SEGMENTO_MODELO].reset_index(drop=True)


# ============================================================================
# 1. VARIABLES DEL MODELO
# ============================================================================
rep.titulo("1. Construcción de las variables del modelo")

X, VARIABLES = construir_variables(df, promedios_municipales=prom, cfg=cfg)


if cfg.LOG_VELOCIDADES:
    FRASE_TRANSFORMACION = (
        "La transformación logarítmica se aplica a las cinco variables que "
        "presentan fuerte asimetría positiva: los accesos, los dos campos de "
        "valores facturados y las dos velocidades.")
else:
    FRASE_TRANSFORMACION = (
        "La transformación logarítmica se aplica a las tres variables de "
        "magnitud que presentan fuerte asimetría positiva: los accesos y los "
        "dos campos de valores facturados. Las dos velocidades se conservan en "
        "su escala original, por las razones que se exponen a continuación.")


rep.tabla(pd.DataFrame({"Variable": VARIABLES}),
          "Variables de entrada del modelo", max_filas=30)

rep.tabla(X.describe().T.round(3).reset_index().rename(columns={"index": "Variable"}),
          "Estadísticos descriptivos de las variables de entrada",
          "Nota. Los máximos de VALOR_POR_ACCESO, RATIO_DOWN_UP, "
          "VEL_RELATIVA_MUNICIPIO y VEL_RELATIVA_TECNOLOGIA corresponden a los "
          "topes definidos en config.py. El acotamiento preserva la condición "
          "extrema del registro: conserva el valor máximo de la variable y el "
          "registro continúa siendo señalado.",
          max_filas=30)

inf.configurar_estilo()

trat_vel = "Logarítmica" if cfg.LOG_VELOCIDADES else "Escala original"
filas_asim = []
for etiqueta, col, trat in [
        ("Cantidad de líneas o accesos", "CANTIDAD_LINEAS_ACCESOS", "Logarítmica"),
        ("Valor facturado o cobrado", "VALOR_FACTURADO_O_COBRADO", "Logarítmica"),
        ("Otros valores facturados", "OTROS_VALORES_FACTURADOS", "Logarítmica"),
        ("Velocidad efectiva de descarga", "VELOCIDAD_EFECTIVA_DOWNSTREAM", trat_vel),
        ("Velocidad efectiva de carga", "VELOCIDAD_EFECTIVA_UPSTREAM", trat_vel)]:
    # La lectura define las velocidades en precisión simple; el cálculo se
    # realiza en doble precisión para que el redondeo no arrastre dígitos
    # espurios a la tabla.
    v = df[col].astype("float64").fillna(0).clip(lower=0)
    filas_asim.append({"Variable": etiqueta,
                       "Asimetría original": round(float(v.skew()), 2),
                       "Asimetría con logaritmo": round(float(np.log1p(v).skew()), 2),
                       "Tratamiento en el modelo": trat})
asim = pd.DataFrame(filas_asim)

rep.tabla(asim, "Efecto de la transformación logarítmica sobre la asimetría",
          "Nota. El coeficiente de asimetría mide el grado de sesgo de la "
          "distribución. Valores próximos a cero indican simetría. El signo "
          "negativo tras la transformación obedece a la elevada proporción de "
          "registros que reportan valor cero, que conservan ese valor y "
          "conforman un grupo diferenciado en el extremo inferior de la escala. "
          "Las dos últimas filas se incluyen con fines comparativos: el "
          "logaritmo corregiría también su asimetría, y la última columna "
          "indica el tratamiento que el modelo aplica efectivamente a cada "
          "variable.")


rep.figura(inf.fig_transformacion_log(df, cfg.DIR_FIG),
           "Distribución de la cantidad de líneas o accesos antes y "
           "después de la transformación logarítmica. El panel izquierdo emplea "
           "escala logarítmica en el eje vertical, dado que la concentración de "
           "registros en el extremo inferior impediría de otro modo distinguir "
           "las frecuencias.")

rep.figura(inf.fig_percentiles(df, cfg, cfg.DIR_FIG, prom),
           "Curva de percentiles de las cuatro variables derivadas que se acotan en la construcción de la matriz de entrada, calculada sobre los valores sin acotar.")


# ============================================================================
# 2. ISOLATION FOREST GLOBAL
# ============================================================================
rep.titulo("2. Isolation Forest global")

det = DetectorAnomalias(n_arboles=cfg.N_ARBOLES, max_samples=cfg.MAX_SAMPLES,
                        contaminacion=cfg.CONTAMINACION, seed=cfg.SEED)

df["SCORE_GLOBAL"] = det.ajustar_global(X)
umbral_g = np.quantile(df["SCORE_GLOBAL"], 1 - cfg.CONTAMINACION)
df["ANOM_GLOBAL"] = df["SCORE_GLOBAL"] >= umbral_g

rep.tabla(pd.DataFrame({
    "Parámetro": ["Número de árboles (t)", "Tamaño de submuestra (ψ)",
                  "Umbral de corte", "Semilla aleatoria",
                  "Puntuación en el umbral", "Registros identificados"],
    "Valor": [cfg.N_ARBOLES, cfg.MAX_SAMPLES,
              f"{cfg.CONTAMINACION*100:.1f}%", cfg.SEED,
              round(umbral_g, 4), f"{df['ANOM_GLOBAL'].sum():,}"],
    "Origen del valor": [
        "Decisión del presente trabajo, a partir del análisis de convergencia",
        "Valor de referencia propuesto por los autores del algoritmo",
        "Criterio operativo del presente trabajo.",
        "Reproducibilidad",
        "Resultado", "Resultado"]}),
    "Configuración y resultado del modelo global",
    "Nota. Únicamente el tamaño de submuestra corresponde al valor de "
    "referencia propuesto por los autores del algoritmo. El número de árboles "
    "se elevó respecto de los 100 que estos proponen, a partir del análisis de "
    "convergencia que se presenta más adelante. El umbral de corte no "
    "interviene en la construcción del modelo, sino únicamente en la conversión "
    "de la puntuación continua en una decisión binaria.")

rep.figura(inf.fig_camino_score(det, X, umbral_g, cfg.DIR_FIG, rng),
           "Relación entre la longitud promedio de camino y la puntuación de anomalía.")



# ============================================================================
# 3. ISOLATION FOREST POR MUNICIPIO
# ============================================================================
rep.titulo("3. Isolation Forest por municipio")

clave = df[cfg.SEGMENTO_LOCAL].astype(str).agg("|".join, axis=1)
score_local, n_modelos, cobertura = det.ajustar_local(X, clave, cfg.MIN_POR_SEGMENTO)
df["SCORE_LOCAL"] = score_local.fillna(df["SCORE_GLOBAL"])
df["SCORE_COMBINADO"] = det.combinar(df["SCORE_GLOBAL"], df["SCORE_LOCAL"],
                                     cfg.PESO_LOCAL)
umbral_c = np.quantile(df["SCORE_COMBINADO"], 1 - cfg.CONTAMINACION)
df["ANOMALIA"] = df["SCORE_COMBINADO"] >= umbral_c


q_loc = np.quantile(df["SCORE_LOCAL"], 1 - cfg.CONTAMINACION)
solo_g = int((df["ANOM_GLOBAL"] & ~(df["SCORE_LOCAL"] >= q_loc)).sum())
solo_l = int((~df["ANOM_GLOBAL"] & (df["SCORE_LOCAL"] >= q_loc)).sum())
ambos = int((df["ANOM_GLOBAL"] & (df["SCORE_LOCAL"] >= q_loc)).sum())

rep.tabla(pd.DataFrame({
    "Criterio": ["Identificados solo por el modelo global",
                 "Identificados solo por los modelos locales",
                 "Identificados por ambos",
                 "Total con la puntuación combinada"],
    "Registros": [solo_g, solo_l, ambos, int(df["ANOMALIA"].sum())],
    "% del total": [round(100*solo_g/len(df), 3), round(100*solo_l/len(df), 3),
                    round(100*ambos/len(df), 3),
                    round(100*df["ANOMALIA"].mean(), 3)]}),
    "Registros identificados según criterio",
    "Nota. Ambos componentes señalan el mismo número de registros por "
    "definición del umbral, de modo que la coincidencia numérica entre las dos "
    "primeras filas es una consecuencia aritmética y no un hallazgo. El dato "
    "informativo es la intersección: la proporción de registros que ambos "
    "componentes señalan simultáneamente. Los registros identificados "
    "únicamente por los modelos locales corresponden a comportamientos que "
    "resultan normales en el agregado nacional pero atípicos dentro de su "
    "municipio.")


filas_peso = []
for p in cfg.PESOS_COMPARACION:
    s = det.combinar(df["SCORE_GLOBAL"], df["SCORE_LOCAL"], p)
    marc = s >= np.quantile(s, 1 - cfg.CONTAMINACION)
    filas_peso.append({
        "Peso local": p,
        "Registros identificados": int(marc.sum()),
        "Coincidencia con el modelo global (%)":
            round(100 * (marc & df["ANOM_GLOBAL"]).sum() / max(marc.sum(), 1), 2)})
rep.tabla(pd.DataFrame(filas_peso),
          "Sensibilidad del resultado a la ponderación")

rep.tabla(comparacion.nucleo_comun_ponderaciones(df, cfg.CONTAMINACION,
                                                 cfg.PESOS_COMPARACION),
          "Núcleo común de registros señalados entre todas las ponderaciones",
          "Nota. El núcleo común reúne los registros que serían señalados con "
          "cualquiera de las ponderaciones evaluadas, incluidas las dos "
          "configuraciones extremas.")


# ============================================================================
# 4. ATRIBUCIÓN POR VARIABLE (DIFFI)
# ============================================================================
rep.titulo("4. Atribución por variable (DIFFI)")

idx_d, idx_a, es_a = det.muestra_diffi(df["ANOMALIA"], cfg.MAX_DIFFI, rng)
imp = det.diffi_global(X.loc[idx_d], es_a)
importancia = (pd.DataFrame({"Variable": VARIABLES, "Importancia": imp.round(4)})
               .sort_values("Importancia", ascending=False).reset_index(drop=True))

rep.tabla(importancia, "Importancia global de las variables",
          f"Nota. Calculada sobre una muestra equilibrada de {len(idx_d):,} "
          f"registros, de los cuales {int(es_a.sum()):,} son señalados y el "
          f"resto normales. La muestra responde al costo computacional: el "
          f"método recorre el camino de cada observación a través de los "
          f"{cfg.N_ARBOLES} árboles. Valores superiores a 1 indican que la "
          f"variable contribuye más a aislar registros anómalos que registros "
          f"normales.", max_filas=25)

# Atribución local sobre la totalidad de los registros señalados
idx_todas = df.index[df["ANOMALIA"]].to_numpy()
loc = det.diffi_local_bloques(X, idx_todas, cfg.BLOQUE_DIFFI)

var_dom = np.array(VARIABLES)[loc.argmax(axis=1)]
peso_dom = loc.max(axis=1)

atrib = (pd.Series(var_dom).value_counts()
         .rename_axis("Variable dominante").reset_index(name="Anomalías"))
atrib["% de las anomalías"] = (100 * atrib["Anomalías"] / atrib["Anomalías"].sum()).round(2)
rep.tabla(atrib, "Variable que explica cada registro anómalo",
          "Nota. Corresponde a la variable con mayor contribución local, "
          "calculada sobre la totalidad de los registros señalados y no sobre "
          "la muestra empleada en la importancia global. Esta atribución "
          "permite indicar al analista qué campo del reporte concentra el "
          "comportamiento inusual, en lugar de entregar una alerta sin "
          "justificación. La importancia global y la atribución local no "
          "coinciden porque miden fenómenos distintos: la primera, la "
          "contribución promedio al aislamiento del conjunto; la segunda, el "
          "campo determinante en cada registro.")


# ============================================================================
# 5. CLASIFICACIÓN SEGÚN LA TIPOLOGÍA
# ============================================================================
rep.titulo("5. Clasificación de las anomalías según la tipología")

anom = df.loc[idx_todas].copy()
anom["VAR_DOMINANTE"] = var_dom
anom["PESO_DOMINANTE"] = peso_dom

cfg.UMBRAL_MULTIVAR = float(np.quantile(peso_dom, cfg.PCT_MULTIVAR))

anom, saltos, det_colectiva, umbral_tasa = clasificar_tipologia(
    df, anom, umbral_g, cfg)

rep.tabla(pd.DataFrame({
    "Tipo": ["Temporal", "Contextual", "Multivariada", "Colectiva", "Puntual"],
    "Criterio de asignación": [
        f"Variación superior al {cfg.UMBRAL_TEMPORAL}% frente al trimestre "
        f"anterior, sobre una base mínima de {cfg.BASE_TEMPORAL} accesos",
        "Alta puntuación en el modelo local y baja en el global",
        f"Ninguna variable concentra más del {cfg.UMBRAL_MULTIVAR*100:.1f}% de "
        f"la contribución local (percentil {cfg.PCT_MULTIVAR:.0%} de la "
        f"distribución observada)",
        f"El proveedor presenta en el trimestre una tasa de señalamiento igual "
        f"o superior a {100*umbral_tasa:.2f}% —{cfg.FACTOR_COLECTIVA} veces la "
        f"tasa general del modelo— con al menos {cfg.MIN_COLECTIVA} registros "
        f"señalados",
        "Alta puntuación global sin cumplir las condiciones anteriores"],
    "Orden de aplicación": [1, 2, 3, 4, 5]}),
    "Criterios de clasificación tipológica",
    "Nota. La categoría colectiva se define como tasa de señalamiento del "
    "grupo proveedor–trimestre y no como conteo absoluto de anomalías: un "
    "criterio de conteo señalaría como colectivos a los proveedores de mayor "
    "tamaño por el solo volumen de registros que reportan, sin que su "
    "diligenciamiento presente una concentración mayor que la del resto. Los "
    "umbrales temporales se establecieron a partir de la exploración de los "
    "datos y del conocimiento del comportamiento de los mercados; admiten "
    "ajuste según el criterio del analista, y su modificación redistribuye los "
    "registros entre categorías sin alterar el conjunto señalado.")

tip = (anom["TIPOLOGIA"].value_counts()
       .rename_axis("Tipo de anomalía").reset_index(name="Registros"))
tip["% del total"] = (100 * tip["Registros"] / tip["Registros"].sum()).round(2)
rep.tabla(tip, "Anomalías identificadas según tipología",
          "Nota. La clasificación se aplica en el orden jerárquico de la tabla "
          "anterior. La distribución entre categorías depende de los umbrales "
          "adoptados; el conjunto de registros señalados, no.")

# ---- Sensibilidad de la tipología a sus umbrales ----
filas_sens = []
for f in (2, 3, 5, 10):
    u = f * (len(anom) / len(df))
    senal = anom.groupby(["ID_EMPRESA", "ORDEN"], observed=True).size()
    total = df.groupby(["ID_EMPRESA", "ORDEN"], observed=True).size()
    tasa = senal / total.reindex(senal.index)
    g = set(tasa[(tasa >= u) & (senal >= cfg.MIN_COLECTIVA)].index)
    n_col = sum((e, o) in g for e, o in zip(anom["ID_EMPRESA"], anom["ORDEN"]))
    filas_sens.append({
        "Factor sobre la tasa general": f,
        "Tasa exigida (%)": round(100 * u, 2),
        "Grupos proveedor–trimestre que la superan": len(g),
        "Registros que cumplen la condición colectiva": n_col,
        "% de los señalados": round(100 * n_col / len(anom), 2)})
rep.tabla(pd.DataFrame(filas_sens),
          "Sensibilidad de la categoría colectiva a su umbral",
          f"Nota. La columna de registros expresa cuántos señalados cumplen la "
          f"condición antes de aplicar la jerarquía, por lo que no coincide con "
          f"el tamaño final de la categoría. El valor adoptado es el factor "
          f"{cfg.FACTOR_COLECTIVA}.")

conc = anom.groupby(["ID_EMPRESA", "EMPRESA", "ORDEN"], observed=True).size()
n_grupos = len(conc)
umbral_conc = conc.quantile(0.93)
grupos_altos = int((conc >= umbral_conc).sum())
anom_altas = int(conc[conc >= umbral_conc].sum())

top = (conc.nlargest(15).reset_index(name="Anomalías")
           .drop(columns="ID_EMPRESA")
           .rename(columns={"EMPRESA": "Proveedor", "ORDEN": "Periodo"}))

rep.tabla(top, "Concentración de anomalías por proveedor y trimestre")

rep.tabla(det_colectiva.head(20).rename(
              columns={"ID_EMPRESA": "Proveedor (NIT)", "ORDEN": "Periodo"}),
          "Tasa de señalamiento por proveedor y trimestre",
          f"Nota. La tasa relaciona los registros señalados con los "
          f"efectivamente reportados por el proveedor en el trimestre, de modo "
          f"que no depende de su tamaño. Se exige una tasa igual o superior a "
          f"{100*umbral_tasa:.2f}% y un mínimo de {cfg.MIN_COLECTIVA} registros "
          f"señalados.", max_filas=20)

rep.tabla(pd.crosstab(anom["TIPOLOGIA"], anom["VAR_DOMINANTE"]).reset_index()
            .rename(columns={"TIPOLOGIA": "Tipo de anomalía"}),
          "Tipología según la variable dominante", max_filas=10)

cols_ej = [c for c in ["PERIODO", "EMPRESA", "MUNICIPIO",
                       "SERVICIO_PAQUETE", "TECNOLOGIA", "CANTIDAD_LINEAS_ACCESOS",
                       "VALOR_FACTURADO_O_COBRADO", "VELOCIDAD_EFECTIVA_DOWNSTREAM",
                       "SCORE_COMBINADO", "VAR_DOMINANTE", "TIPOLOGIA"]
           if c in anom.columns]
rep.tabla(anom.nlargest(20, "SCORE_COMBINADO")[cols_ej],
          "Registros con mayor puntuación de anomalía", max_filas=20)

# ---- Contraste ilustrativo entre extremos ----

COLS = [c for c in ["PERIODO", "MUNICIPIO", "TECNOLOGIA", "CANTIDAD_LINEAS_ACCESOS",
                    "VELOCIDAD_EFECTIVA_DOWNSTREAM", "VELOCIDAD_EFECTIVA_UPSTREAM",
                    "VALOR_FACTURADO_O_COBRADO", "SCORE_COMBINADO"]
        if c in df.columns]

RENOM = {"PERIODO": "Periodo", "MUNICIPIO": "Municipio",
         "TECNOLOGIA": "Tecnología", "CANTIDAD_LINEAS_ACCESOS": "Accesos",
         "VELOCIDAD_EFECTIVA_DOWNSTREAM": "Descarga (Mbps)",
         "VELOCIDAD_EFECTIVA_UPSTREAM": "Carga (Mbps)",
         "VALOR_FACTURADO_O_COBRADO": "Facturado (COP)",
         "SCORE_COMBINADO": "Puntuación"}

extremos = pd.concat([
    df.nlargest(5, "SCORE_COMBINADO")[COLS].assign(Grupo="Señalado"),
    df.nsmallest(5, "SCORE_COMBINADO")[COLS].assign(Grupo="No señalado"),
])
extremos["Valor por acceso"] = (
    extremos["VALOR_FACTURADO_O_COBRADO"]
    / extremos["CANTIDAD_LINEAS_ACCESOS"].replace(0, np.nan)).round(0)
extremos = (extremos.rename(columns=RENOM)
            [["Grupo"] + [RENOM[c] for c in COLS] + ["Valor por acceso"]]
            .reset_index(drop=True))

rep.tabla(extremos, "Registros con puntuación extrema en ambos grupos",
          "Nota. Los cinco primeros corresponden a la puntuación más alta y los "
          "cinco restantes a la más baja. La columna final presenta el valor "
          "facturado por acceso, magnitud derivada que no forma parte del "
          "reporte.")


# ============================================================================
# 6. MÉTRICAS DE EVALUACIÓN DEL MODELO
# ============================================================================
rep.titulo("6. Métricas de evaluación del modelo")

# ---------------------------------------------------------------------------
rep.titulo("6.1 Caracterización de la puntuación", 3)

rep.tabla(metricas.descriptivos_puntuacion(df, cfg.CONTAMINACION),
          "Estadísticos de las puntuaciones de anomalía",
          "Nota. La puntuación global y la local se expresan en la escala del "
          "algoritmo, en la que 0,5 corresponde a la ausencia de anomalías "
          "distinguibles; la combinada resulta de normalizar ambas al intervalo "
          "unitario y ponderarlas. El umbral de corte corresponde al percentil "
          "99 de cada distribución.")

rep.tabla(metricas.separacion(df, cfg.CONTAMINACION, seed=cfg.SEED),
          "Separación entre los registros señalados y el resto",
          "Nota. La diferencia estandarizada expresa cuántas desviaciones "
          "típicas separan la puntuación media de ambos grupos. El valor no se "
          "interpreta frente al umbral convencional de 0,8, que corresponde a "
          "la comparación de dos grupos definidos con independencia de la "
          "variable medida: aquí los grupos se construyen cortando la propia "
          "distribución de la puntuación, operación que produce una diferencia "
          "elevada por construcción. La referencia es la última columna: el "
          "cociente entre la diferencia observada y la que el mismo corte "
          "induce sobre una distribución unimodal sin anomalías.")

rep.tabla(metricas.referencia_separacion(cfg.CONTAMINACION, seed=cfg.SEED),
          "Diferencia estandarizada inducida por el corte en distribuciones "
          "sin anomalías",
          f"Nota. Cada fila corresponde a una muestra simulada de una "
          f"distribución unimodal continua, cortada en el percentil "
          f"{100*(1-cfg.CONTAMINACION):.0f}.")

rep.tabla(metricas.forma_de_la_cola(df, cfg.CONTAMINACION),
          "Forma de la distribución en el entorno del umbral",
          "Nota. Si el conjunto señalado constituyera un grupo desprendido del "
          "resto, existiría entre el último registro no señalado y el primero "
          "señalado un salto apreciablemente mayor que la distancia típica "
          "entre registros contiguos de esa misma zona de la distribución. La "
          "comparación pertinente es con la distancia típica en el 2% "
          "superior y no con la de la zona central, dado que en el centro la "
          "densidad de registros es mucho mayor y cualquier separación de la "
          "cola resultaría elevada frente a ella.")

rep.figura(inf.fig_cola_puntuacion(df, umbral_c, cfg.DIR_FIG),
           "Extremo superior de la distribución de la puntuación combinada.")

# ---------------------------------------------------------------------------
rep.titulo("6.2 Estabilidad del modelo global", 3)

est_sem, _, punt_sem = metricas.estabilidad_semillas(
    X, cfg.CONTAMINACION, cfg.N_ARBOLES, cfg.MAX_SAMPLES,
    semillas=cfg.SEMILLAS_ESTABILIDAD, rng=rng)
rep.tabla(est_sem, "Estabilidad ante distintas semillas aleatorias",
          "Nota. Calculada sobre una submuestra aleatoria del conjunto, con la "
          "configuración definitiva del modelo.")

rep.tabla(metricas.validez_equivalente(punt_sem, cfg.CONTAMINACION),
          "Ubicación de los registros que varían al cambiar la semilla",
          "Nota. La estabilidad imperfecta del conjunto señalado solo autoriza "
          "a describir las listas alternativas como de validez equivalente si "
          "los registros que entran y salen son los situados en la frontera del "
          "umbral. Esta tabla verifica esa condición: si los registros "
          "intercambiados estuvieran repartidos por toda la distribución, el "
          "resultado sería inestable en sentido fuerte y las listas no serían "
          "intercambiables.")

rep.tabla(metricas.convergencia_arboles(X, cfg.CONTAMINACION,
                                        tamanos=cfg.ARBOLES_CONVERGENCIA,
                                        max_samples=cfg.MAX_SAMPLES,
                                        seed=cfg.SEED, rng=rng),
          "Convergencia según el número de árboles",
          f"Nota. Se toma como referencia el bosque de "
          f"{max(cfg.ARBOLES_CONVERGENCIA)} árboles, superior al adoptado en la "
          f"configuración definitiva ({cfg.N_ARBOLES}); de otro modo la última "
          f"fila valdría 1 por construcción y no informaría sobre la "
          f"convergencia. La correlación se estabiliza antes que la coincidencia "
          f"entre conjuntos señalados, porque el ordenamiento general converge "
          f"con rapidez mientras que la decisión en el entorno del umbral "
          f"involucra registros con puntuaciones casi idénticas.")

# ---------------------------------------------------------------------------
rep.titulo("6.3 Componente segmentado", 3)

rep.tabla(metricas.cobertura_segmentos(df, clave, cfg.MIN_POR_SEGMENTO),
          "Cobertura de los subespacios",
          "Nota. Los subespacios que no alcanzan el tamaño mínimo conservan la "
          "puntuación del modelo global.")

rep.tabla(metricas.concordancia_componentes(df, cfg.CONTAMINACION, rng=rng),
          "Concordancia entre el componente global y el segmentado",
          "Nota. Una correlación alta en el conjunto de la distribución junto "
          "con una coincidencia baja entre los conjuntos señalados indica que "
          "ambos componentes ordenan los registros de forma similar en el "
          "agregado, pero difieren precisamente en el extremo superior, que es "
          "donde se toma la decisión de señalamiento.")

# ---------------------------------------------------------------------------
rep.titulo("6.4 Estabilidad de la atribución por variable", 3)

est_diffi, det_diffi = metricas.estabilidad_diffi(
    det, X, df["ANOMALIA"], VARIABLES, repeticiones=5,
    max_registros=cfg.MAX_DIFFI, rng=rng)

rep.tabla(est_diffi, "Estabilidad de la importancia global",
          "Nota. El índice de Herfindahl mide la concentración de la importancia: "
          "valores próximos a cero indican una contribución repartida entre "
          "muchas variables; valores altos, dependencia de unas pocas.")

rep.tabla(det_diffi, "Importancia media y variabilidad por variable",
          "Nota. El coeficiente de variación expresa la dispersión de la "
          "importancia entre repeticiones, en términos relativos a su media.",
          max_filas=25)

rep.tabla(metricas.dispersion_atribucion(anom["PESO_DOMINANTE"],
                                         cfg.UMBRAL_MULTIVAR),
          "Dispersión de la atribución en los registros señalados",
          "Nota. Los registros con atribución dispersa corresponden a "
          "comportamientos que no se explican por una única variable, y se "
          "clasifican como anomalías multivariadas.")

# ---------------------------------------------------------------------------
rep.titulo("6.5 Complementariedad frente a las reglas de validación", 3)

t_contraste, incumple = metricas.contraste_reglas(df)

rep.tabla(t_contraste, "Contraste entre el modelo y las reglas vigentes",
          "Nota. Las reglas de primer nivel comprenden los catálogos de segmento, "
          "servicio, tecnología y estado; los rangos admisibles de accesos y "
          "valores facturados; la correspondencia entre servicio, velocidad y "
          "tecnología; y las condiciones aplicables al segmento 117.")

rep.tabla(metricas.contraste_observado_esperado(df, incumple),
          "Solapamiento observado frente al esperado bajo independencia",
          "Nota. La proporción de registros señalados que cumple todas las "
          "reglas no constituye por sí sola una medida del aporte del modelo: "
          "dado que solo una fracción reducida de la base incumple alguna "
          "regla, cualquier selección de registros —incluida una selección "
          "aleatoria— arrojaría una proporción de cumplimiento igualmente alta. "
          "El indicador informativo es la comparación entre el solapamiento "
          "observado y el esperado bajo independencia, cuya distribución es "
          "hipergeométrica. Un solapamiento significativamente inferior al "
          "esperado acredita que ambos mecanismos operan sobre conjuntos "
          "prácticamente disjuntos.")

rep.tabla(metricas.contraste_por_tipologia(anom, incumple),
          "Cumplimiento de las reglas según el tipo de anomalía",
          "Nota. Permite establecer qué tipos de comportamiento inusual escapan "
          "en mayor medida a los mecanismos de verificación vigentes.")


# ============================================================================
# 7. COMPARACIÓN DE CONFIGURACIONES
# ============================================================================
rep.titulo("7. Comparación de configuraciones del modelo")

# ---------------------------------------------------------------------------
rep.titulo("7.1 Ponderación entre el componente global y el segmentado", 3)

t_pond = comparacion.comparar_ponderaciones(df, cfg.CONTAMINACION,
                                            pesos=cfg.PESOS_COMPARACION,
                                            incumple=incumple)
rep.tabla(t_pond, "Comparación según la ponderación aplicada",
          "Nota. Un peso local de cero corresponde al modelo global; un peso de "
          "uno, exclusivamente a los modelos por subespacio. La coincidencia se "
          "calcula frente al conjunto señalado por el modelo global. La última "
          "columna expresa la proporción de registros identificados que no "
          "habría sido detectada por las reglas de validación vigentes; su "
          "lectura requiere compararla con la tasa de incumplimiento de la "
          "base, presentada en el apartado anterior.")

# ---------------------------------------------------------------------------
rep.titulo("7.2 Umbral de corte", 3)

t_umb = comparacion.comparar_umbrales(df, umbrales=cfg.UMBRALES_COMPARACION,
                                      incumple=incumple)
rep.tabla(t_umb, "Efecto del umbral de corte sobre el volumen señalado",
          "Nota. El promedio por trimestre permite dimensionar la carga de "
          "revisión que implicaría cada umbral. Las columnas de coincidencia "
          "indican cuántos registros del conjunto adoptado se conservan al "
          "modificar la proporción de contaminación: los conjuntos están "
          "anidados por construcción, de modo que reducir el umbral a la mitad "
          "retiene la mitad superior del conjunto actual y ampliarlo lo "
          "contiene íntegramente.")

# ---------------------------------------------------------------------------
rep.titulo("7.3 Criterio de segmentación", 3)

ESQUEMAS = {
    "Departamento": ["ID_DEPARTAMENTO"],
    "Municipio": ["ID_MUNICIPIO"],
    "Municipio y tecnología": ["ID_MUNICIPIO", "ID_TECNOLOGIA"],
    "Municipio y servicio": ["ID_MUNICIPIO", "ID_SERVICIO_PAQUETE"],
}

t_seg = comparacion.comparar_segmentaciones(
    df, X, det, cfg.CONTAMINACION, cfg.PESO_LOCAL, ESQUEMAS,
    cfg.MIN_POR_SEGMENTO, incumple=incumple)
rep.tabla(t_seg, "Comparación según el criterio de segmentación",
          f"Nota. Todos los esquemas exigen un mínimo de "
          f"{cfg.MIN_POR_SEGMENTO} registros por subespacio; los que no lo "
          f"alcanzan conservan la puntuación global. La cobertura indica el "
          f"porcentaje de registros efectivamente evaluados por un modelo local. "
          f"El esquema que incorpora la tecnología es el pertinente para el "
          f"tratamiento de los proveedores cuya oferta difiere de la "
          f"predominante en el territorio.")


# ============================================================================
# 8. FIGURAS DE RESULTADOS
# ============================================================================
rep.titulo("8. Figuras de resultados")

rep.figura(inf.fig_distribucion(df, umbral_c, cfg.CONTAMINACION, cfg.DIR_FIG),
           "Distribución de la puntuación combinada de anomalía. La "
           "escala vertical es logarítmica.")
rep.figura(inf.fig_por_trimestre(df, cfg.DIR_FIG),
           "Proporción de registros anómalos por trimestre.")
rep.figura(inf.fig_global_local(df, cfg.DIR_FIG, rng),
           "Relación entre la puntuación global y la puntuación local. "
           "Los registros situados en la parte superior izquierda corresponden a "
           "anomalías de carácter contextual.")
rep.figura(inf.fig_diffi(importancia, cfg.DIR_FIG),
           "Importancia global de las variables según DIFFI. La línea "
           "vertical marca el valor de referencia.")
rep.figura(inf.fig_tipologia(tip, cfg.DIR_FIG),
           "Distribución de las anomalías según la tipología establecida.")


# ============================================================================
# 9. EXPORTACIÓN
# ============================================================================
import os
os.makedirs(cfg.DIR_SALIDA, exist_ok=True)
anom.to_csv(cfg.SALIDA_CSV, index=False, encoding="utf-8-sig")
rep.guardar(cfg.SALIDA_MODELO, pie=inf.pie_de_ejecucion(cfg, len(df)))

print(f"Figuras en: {cfg.DIR_FIG}/")
print(f"Anomalías exportadas: {cfg.SALIDA_CSV} ({len(anom):,} registros)")
