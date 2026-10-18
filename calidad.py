"""
calidad.py — Evaluación de las dimensiones de calidad de datos
Reporte T.1.3 de servicios fijos · CRC

Las verificaciones se derivan de las reglas de validación de primer nivel
definidas en el diccionario oficial del formato, conforme a lo establecido
en la Resolución CRC 5050 de 2016.

Nota sobre la especificación: el formato no admite campos vacíos. Los campos
que no aplican a un servicio se diligencian con valores convencionales (cero
para las velocidades y 999 para la tecnología).
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- CATÁLOGOS
CAT_SEGMENTO   = [101, 102, 103, 104, 105, 106, 107, 108, 109, 117]
CAT_SERVICIO   = [1, 2, 3, 4, 5, 6, 7]
CAT_TECNOLOGIA = list(range(101, 116)) + [999]   # el 0 no pertenece al catálogo
CAT_ESTADO     = [1, 2]

SERVICIOS_INTERNET     = [1, 4, 5, 7]   # exigen velocidad > 0 y tecnología real
SERVICIOS_SIN_INTERNET = [2, 3, 6]      # exigen velocidad = 0 y tecnología 999

TECNOLOGIA_NA = 999
SEGMENTO_ACCESOS_ADICIONALES = 117      # exige facturación en cero

# Campos definidos en el diccionario del formato
CAMPOS_FORMATO = [
    "ANNO", "TRIMESTRE", "ID_EMPRESA", "ID_MUNICIPIO", "ID_DEPARTAMENTO",
    "ID_SEGMENTO", "ID_SERVICIO_PAQUETE", "ID_TECNOLOGIA", "ID_ESTADO",
    "VELOCIDAD_EFECTIVA_DOWNSTREAM", "VELOCIDAD_EFECTIVA_UPSTREAM",
    "CANTIDAD_LINEAS_ACCESOS", "VALOR_FACTURADO_O_COBRADO",
    "OTROS_VALORES_FACTURADOS",
]

# Columnas descriptivas que la publicación agrega como etiqueta de los códigos
DESCRIPTIVAS = ["EMPRESA", "MUNICIPIO", "DEPARTAMENTO", "SEGMENTO",
                "SERVICIO_PAQUETE", "TECNOLOGIA", "ESTADO"]

# Campos que definen la unicidad del registro según el formato.
LLAVE_PRIMARIA = ["ANNO", "TRIMESTRE", "ID_EMPRESA", "ID_MUNICIPIO",
                  "ID_SEGMENTO", "ID_SERVICIO_PAQUETE",
                  "VELOCIDAD_EFECTIVA_DOWNSTREAM", "VELOCIDAD_EFECTIVA_UPSTREAM",
                  "ID_TECNOLOGIA", "ID_ESTADO"]

# ============================================================================
# COMPLETITUD
# ============================================================================
def completitud(df):
    """
    Completitud sobre los campos definidos en el diccionario del formato.

    Se excluyen las columnas descriptivas que la publicación agrega como
    etiqueta de los códigos, por no formar parte de la especificación del
    reporte ni ser diligenciadas por los proveedores.
    """
    filas = []
    for c in CAMPOS_FORMATO:
        if c not in df.columns:
            continue
        nulos = int(df[c].isna().sum())
        filas.append({
            "Variable": c,
            "Obligatorio": "Sí",
            "Sin informar": nulos,
            "Completitud (%)": round(100 * (1 - nulos / len(df)), 3),
        })

    t = (pd.DataFrame(filas).sort_values("Completitud (%)")
         .reset_index(drop=True))

    # El indicador se calcula sobre registros distintos con al menos un campo
    # sin informar, de modo que sea homogéneo con el de las demás dimensiones
    # y no dependa del número de campos evaluados.
    campos = [c for c in CAMPOS_FORMATO if c in df.columns]
    incompleto = df[campos].isna().any(axis=1)
    indicador = 100 * (1 - incompleto.sum() / len(df))

    t.loc[len(t)] = ["Registros distintos con al menos un campo sin informar",
                     "", int(incompleto.sum()), round(indicador, 3)]
    return t, indicador


def cobertura_descriptivas(df):
    """Cobertura de las etiquetas que la publicación asocia a cada código."""
    filas = []
    for c in DESCRIPTIVAS:
        if c not in df.columns:
            continue
        nulos = int(df[c].isna().sum())
        filas.append({
            "Columna descriptiva": c,
            "Sin etiqueta": nulos,
            "Cobertura (%)": round(100 * (1 - nulos / len(df)), 3),
        })
    return pd.DataFrame(filas).sort_values("Cobertura (%)")


# ============================================================================
# EXACTITUD — dominio admisible de cada campo
# ============================================================================
def exactitud(df, excluir_convencion=True):
    """
    Valores fuera de la regla del diccionario definido para cada campo.

    excluir_convencion : si es True, el indicador global excluye los registros
        de servicios sin internet que reportan el código cero en tecnología,
        por tratarse de una regla del diccionario generalizada de la base publicada y no
        de un incumplimiento individual. La verificación se conserva en la
        tabla con fines informativos.
    """
    es_int = df["ID_SERVICIO_PAQUETE"].isin(SERVICIOS_INTERNET)
    no_int = df["ID_SERVICIO_PAQUETE"].isin(SERVICIOS_SIN_INTERNET)

    mascaras = {
        "Trimestre fuera del rango 1–4":
            ~df["TRIMESTRE"].between(1, 4),
        "Municipio sin codificación DIVIPOLA":
            df["ID_MUNICIPIO"].isna() | (df["ID_MUNICIPIO"] <= 0),
        "Segmento fuera de la lista de referencia (101–109 y 117)":
            ~df["ID_SEGMENTO"].isin(CAT_SEGMENTO),
        "Servicio fuera de la lista de referencia (1–7)":
            ~df["ID_SERVICIO_PAQUETE"].isin(CAT_SERVICIO),
        "Tecnología fuera de la lista de referencia (101–115 y 999)":
            ~df["ID_TECNOLOGIA"].isin(CAT_TECNOLOGIA) & (df["ID_TECNOLOGIA"] != 0),
        "Tecnología en cero en servicios sin internet":
            no_int & (df["ID_TECNOLOGIA"] == 0),
        "Tecnología en cero en servicios con internet":
            es_int & (df["ID_TECNOLOGIA"] == 0),
        "Estado fuera de la lista de referencia (1–2)":
            ~df["ID_ESTADO"].isin(CAT_ESTADO),
        "Accesos no mayores que cero":
            df["CANTIDAD_LINEAS_ACCESOS"] <= 0,
        "Valor facturado negativo":
            df["VALOR_FACTURADO_O_COBRADO"] < 0,
        "Otros valores facturados negativos":
            df["OTROS_VALORES_FACTURADOS"] < 0,
        "Valor facturado con decimales":
            df["VALOR_FACTURADO_O_COBRADO"] % 1 != 0,
        "Velocidad con más de tres decimales":
            (df["VELOCIDAD_EFECTIVA_DOWNSTREAM"] * 1000) % 1 != 0,
    }

    t = pd.DataFrame([(k, int(v.sum())) for k, v in mascaras.items()],
                     columns=["Verificación", "Registros"])
    t["% del total"] = (100 * t["Registros"] / len(df)).round(2)

    # Registros distintos que fallan al menos una verificación
    clave_conv = "Tecnología en cero en servicios sin internet"
    usadas = {k: v for k, v in mascaras.items()
              if not (excluir_convencion and k == clave_conv)}
    alguna = pd.concat(usadas.values(), axis=1).any(axis=1)
    indicador = 100 * (1 - alguna.sum() / len(df))

    # Indicador sin exclusiones, para efectos comparativos
    todas = pd.concat(mascaras.values(), axis=1).any(axis=1)
    indicador_total = 100 * (1 - todas.sum() / len(df))

    # Totales de registros distintos: son el numerador del indicador y
    # permiten reproducir el cálculo a partir de la tabla. No coinciden con
    # la suma de las filas anteriores, porque un mismo registro puede
    # incumplir varias verificaciones y se contabiliza una sola vez.
    t = t.reset_index(drop=True)
    t.loc[len(t)] = ["Registros distintos que incumplen al menos una "
                     "verificación",
                     int(alguna.sum()), round(100 * alguna.sum() / len(df), 2)]
    t.loc[len(t)] = ["Registros distintos que incumplen al menos una "
                     "verificación (se incluye el reporte de tecnología "
                     "en cero en servicios sin internet)",
                     int(todas.sum()), round(100 * todas.sum() / len(df), 2)]

    return t, indicador, indicador_total


def exactitud_velocidades(df):
    """
    Rangos de velocidad por segmento, sobre los servicios que incluyen
    internet fijo. Los servicios sin internet reportan velocidad cero por
    especificación y quedan excluidos de esta lectura.
    """
    inter = df[df["ES_INTERNET"] & (df["VELOCIDAD_EFECTIVA_DOWNSTREAM"] > 0)]
    filas = []
    for seg in ["Residencial", "Corporativo"]:
        d = inter[inter["SEG_GRUPO"] == seg]
        if len(d) == 0:
            continue
        v = d["VELOCIDAD_EFECTIVA_DOWNSTREAM"]
        filas.append({
            "Segmento": seg,
            "Registros": len(d),
            "Mínima (Mbps)": round(v.min(), 3),
            "Mediana (Mbps)": round(v.median(), 2),
            "P95 (Mbps)": round(v.quantile(0.95), 2),
            "P99.9 (Mbps)": round(v.quantile(0.999), 2),
            "Máxima (Mbps)": round(v.max(), 2),
            "Sobre P99.9": int((v > v.quantile(0.999)).sum()),
            "Menores a 1 Mbps": int((v < 1).sum()),
        })
    return pd.DataFrame(filas)


def tecnologia_en_cero(df):
    """
    Detalle de los registros que reportan el código cero en tecnología,
    desagregados por servicio y segmento.

    El código cero no pertenece al catálogo definido en el formato, que
    contempla los valores 101 a 115 para las tecnologías de acceso y 999
    para los servicios que no incluyen internet fijo.
    """
    cero = df[df["ID_TECNOLOGIA"] == 0]
    if cero.empty:
        return pd.DataFrame()

    t = (cero.groupby(["ID_SERVICIO_PAQUETE", "SERVICIO_PAQUETE", "SEG_GRUPO"],
                      observed=True)
             .agg(Registros=("ANNO", "size"),
                  Proveedores=("ID_EMPRESA", "nunique"),
                  Municipios=("ID_MUNICIPIO", "nunique"),
                  Trimestres=("ORDEN", "nunique"))
             .reset_index())

    t["Valor esperado"] = np.where(
        t["ID_SERVICIO_PAQUETE"].isin(SERVICIOS_INTERNET),
        "Tecnología 101–115", "Código 999")

    t["ID_SERVICIO_PAQUETE"] = t["ID_SERVICIO_PAQUETE"].astype(int)

    return (t.sort_values(["ID_SERVICIO_PAQUETE", "SEG_GRUPO"])
             .rename(columns={"ID_SERVICIO_PAQUETE": "Código",
                              "SERVICIO_PAQUETE": "Servicio o paquete",
                              "SEG_GRUPO": "Segmento"}))

# ============================================================================
# CONSISTENCIA — coherencia entre campos
# ============================================================================
def consistencia(df, excluir_convencion=True):
    """
    Reglas de coherencia entre campos, según la especificación del formato.

    Se verifican tres bloques: la correspondencia entre el servicio reportado
    y los campos de velocidad y tecnología; las condiciones aplicables al
    segmento de accesos adicionales (117); y la unicidad de los registros.

    El indicador cuenta registros distintos que incumplen al menos una regla,
    de modo que un mismo registro no se penaliza más de una vez.

    La condición de que la velocidad de carga no supere a la de descarga se
    incorpora en este trabajo: el diccionario del formato no establece
    relación alguna entre ambos campos.
    """
    es_int = df["ID_SERVICIO_PAQUETE"].isin(SERVICIOS_INTERNET)
    no_int = df["ID_SERVICIO_PAQUETE"].isin(SERVICIOS_SIN_INTERNET)
    es_117 = df["ID_SEGMENTO"] == SEGMENTO_ACCESOS_ADICIONALES

    llaves = [c for c in LLAVE_PRIMARIA if c in df.columns]

    mascaras = {
        # --- Servicio frente a velocidad ---
        "Servicio con internet y velocidad de descarga en cero":
            es_int & (df["VELOCIDAD_EFECTIVA_DOWNSTREAM"] <= 0),
        "Servicio con internet y velocidad de carga en cero":
            es_int & (df["VELOCIDAD_EFECTIVA_UPSTREAM"] <= 0),
        "Servicio sin internet y velocidad de descarga distinta de cero":
            no_int & (df["VELOCIDAD_EFECTIVA_DOWNSTREAM"] != 0),
        "Servicio sin internet y velocidad de carga distinta de cero":
            no_int & (df["VELOCIDAD_EFECTIVA_UPSTREAM"] != 0),

        # --- Servicio frente a tecnología ---
        "Servicio con internet y tecnología igual a 999":
            es_int & (df["ID_TECNOLOGIA"] == TECNOLOGIA_NA),
        "Servicio con internet y tecnología en cero":
            es_int & (df["ID_TECNOLOGIA"] == 0),
        "Servicio sin internet y tecnología en cero":
            no_int & (df["ID_TECNOLOGIA"] == 0),
        "Servicio sin internet y tecnología de acceso reportada (101–115)":
            no_int & df["ID_TECNOLOGIA"].isin(range(101, 116)),

        # --- Segmento 117: accesos adicionales corporativos ---
        "Segmento 117 con servicio distinto de 1, 2 o 3":
            es_117 & ~df["ID_SERVICIO_PAQUETE"].isin([1, 2, 3]),
        "Segmento 117 con valor facturado distinto de cero":
            es_117 & (df["VALOR_FACTURADO_O_COBRADO"] != 0),
        "Segmento 117 con otros valores distintos de cero":
            es_117 & (df["OTROS_VALORES_FACTURADOS"] != 0),

        # --- Coherencia interna adicional ---
        "Velocidad de carga mayor que la de descarga":
            df["VELOCIDAD_EFECTIVA_UPSTREAM"] > df["VELOCIDAD_EFECTIVA_DOWNSTREAM"],

        # --- Unicidad ---
        "Registros duplicados por llave primaria":
            df.duplicated(subset=llaves, keep=False) & df.duplicated(subset=llaves),
    }

    t = pd.DataFrame([(k, int(v.sum())) for k, v in mascaras.items()],
                     columns=["Verificación", "Registros"])
    t["% del total"] = (100 * t["Registros"] / len(df)).round(2)

    clave_conv = "Servicio sin internet y tecnología en cero"
    usadas = {k: v for k, v in mascaras.items()
              if not (excluir_convencion and k == clave_conv)}
    alguna = pd.concat(usadas.values(), axis=1).any(axis=1)
    indicador = 100 * (1 - alguna.sum() / len(df))

    todas = pd.concat(mascaras.values(), axis=1).any(axis=1)
    indicador_total = 100 * (1 - todas.sum() / len(df))

    t = t.reset_index(drop=True)
    t.loc[len(t)] = ["Registros distintos que incumplen al menos una regla",
                     int(alguna.sum()), round(100 * alguna.sum() / len(df), 2)]
    t.loc[len(t)] = ["Registros distintos que incumplen al menos una regla "
                     "(verificación estricta)",
                     int(todas.sum()), round(100 * todas.sum() / len(df), 2)]

    return t, indicador, indicador_total


# ============================================================================
# SIMETRÍA ENTRE LAS DOS VELOCIDADES
# ============================================================================
# Tecnologías de acceso capaces de entregar velocidades simétricas: las que
# llevan fibra hasta el domicilio, el predio o el edificio, donde la
# distribución interna se realiza sobre medios simétricos. Los accesos que
# conservan un tramo final de cobre o de radio —xDSL, FTTC, FTTN, HFC, cable,
# satelital e inalámbricos— son asimétricos por diseño. La categoría de otras
# tecnologías de fibra se trata por separado, porque agrupa accesos de
# arquitectura diversa y no admite una clasificación unívoca.
TECNOLOGIAS_SIMETRICAS = (
    "Fiber to the home (FTTH)",
    "Fiber to the premises",
    "Fiber to the building o fiber to the basement (FTTB)",
)
TECNOLOGIAS_INDETERMINADAS = ("Otras tecnologías de fibra (antes FTTx)",)


def simetria_velocidades(df, segmento=None, min_registros=100):
    """
    Registros que declaran la misma velocidad de descarga y de carga.

    La igualdad entre ambos campos no incumple ninguna regla de primer nivel:
    la especificación solo prohíbe que la velocidad de carga supere a la de
    descarga. La verificación es, por tanto, un análisis de plausibilidad y no
    una regla del formato, y no interviene en los indicadores de las cuatro
    dimensiones.

    segmento : si se indica, restringe el análisis a ese valor de SEG_GRUPO.
        Se emplea para que la base coincida con la del modelo y las cifras no
        se mezclen con las del agregado de los dos segmentos.

    Devuelve (resumen, por_tecnologia, por_proveedor), donde por_proveedor
    desagrega la tecnología xDSL, asimétrica por definición, entre los
    proveedores con al menos min_registros observaciones en ella.
    """
    if segmento:
        df = df[df["SEG_GRUPO"] == segmento]

    base = df[df["ES_INTERNET"]
              & (df["VELOCIDAD_EFECTIVA_DOWNSTREAM"] > 0)
              & (df["VELOCIDAD_EFECTIVA_UPSTREAM"] > 0)].copy()
    base["SIMETRICO"] = (base["VELOCIDAD_EFECTIVA_DOWNSTREAM"]
                         == base["VELOCIDAD_EFECTIVA_UPSTREAM"])

    tec = base["TECNOLOGIA"].astype(str)
    simetrica = tec.isin(TECNOLOGIAS_SIMETRICAS)
    indeterminada = tec.isin(TECNOLOGIAS_INDETERMINADAS)
    asimetrica = ~simetrica & ~indeterminada

    n = len(base)
    n_sim = int(base["SIMETRICO"].sum())
    n_impl = int((base["SIMETRICO"] & asimetrica).sum())

    resumen = pd.DataFrame({
        "Indicador": [
            f"Registros{' del segmento ' + segmento.lower() if segmento else ''} "
            f"con internet fijo y ambas velocidades positivas",
            "Registros con velocidades idénticas",
            "Proporción sobre los registros evaluados (%)",
            "De ellos, en tecnologías capaces de entregar simetría",
            "De ellos, en tecnologías asimétricas por diseño",
            "Proporción de estos últimos sobre los registros evaluados (%)"],
        "Valor": [
            f"{n:,}", f"{n_sim:,}", round(100 * n_sim / max(n, 1), 2),
            f"{int((base['SIMETRICO'] & simetrica).sum()):,}",
            f"{n_impl:,}", round(100 * n_impl / max(n, 1), 2)]})

    por_tecnologia = (base.groupby("TECNOLOGIA", observed=True)["SIMETRICO"]
                      .agg(Registros="size", Simétricos="sum")
                      .reset_index())
    por_tecnologia["% simétricos"] = (
        100 * por_tecnologia["Simétricos"] / por_tecnologia["Registros"]).round(1)
    por_tecnologia["Simetría técnicamente viable"] = np.where(
        por_tecnologia["TECNOLOGIA"].astype(str).isin(TECNOLOGIAS_SIMETRICAS), "Sí",
        np.where(por_tecnologia["TECNOLOGIA"].astype(str)
                 .isin(TECNOLOGIAS_INDETERMINADAS), "Indeterminada", "No"))
    por_tecnologia = (por_tecnologia
                      .rename(columns={"TECNOLOGIA": "Tecnología"})
                      .sort_values("Registros", ascending=False)
                      .reset_index(drop=True))

    xdsl = base[tec.str.contains("xDSL", case=False, na=False)]
    por_proveedor = (xdsl.groupby("EMPRESA", observed=True)["SIMETRICO"]
                     .agg(Registros="size", Simétricos="sum")
                     .reset_index())
    por_proveedor = por_proveedor[por_proveedor["Registros"] >= min_registros]
    por_proveedor["% simétricos"] = (
        100 * por_proveedor["Simétricos"] / por_proveedor["Registros"]).round(1)
    por_proveedor = (por_proveedor.rename(columns={"EMPRESA": "Proveedor"})
                     .sort_values("Registros", ascending=False)
                     .reset_index(drop=True))

    return resumen, por_tecnologia, por_proveedor


def velocidad_vs_promedio(df, promedios, factor=3):
    """
    Velocidad reportada frente al promedio ponderado de su municipio,
    segmento y trimestre.

    Esta verificación no forma parte de las reglas de primer nivel del formato;
    corresponde a un análisis de plausibilidad en contexto territorial.

    Devuelve (base con la razón calculada, resumen por segmento).
    """
    claves = ["ANNO", "TRIMESTRE", "ID_MUNICIPIO", "SEG_GRUPO"]

    base = df[df["ES_INTERNET"] & (df["VELOCIDAD_EFECTIVA_DOWNSTREAM"] > 0)].copy()
    base = base.merge(promedios[claves + ["VEL_PROMEDIO"]], on=claves, how="left")
    base["RAZON"] = (base["VELOCIDAD_EFECTIVA_DOWNSTREAM"]
                     / base["VEL_PROMEDIO"].replace(0, np.nan))

    filas = []
    for seg in ["Residencial", "Corporativo"]:
        d = base[base["SEG_GRUPO"] == seg]
        if len(d) == 0:
            continue
        filas.append({
            "Segmento": seg,
            "Registros evaluados": len(d),
            "Velocidad promedio (Mbps)": round(d["VELOCIDAD_EFECTIVA_DOWNSTREAM"].mean(), 2),
            "Sobre el promedio": int((d["RAZON"] > 1).sum()),
            "% sobre el promedio": round(100 * (d["RAZON"] > 1).mean(), 2),
            f"Más de {factor}x el promedio": int((d["RAZON"] > factor).sum()),
            f"Proveedores > {factor}x": int(d.loc[d["RAZON"] > factor, "ID_EMPRESA"].nunique()),
            "Municipios afectados": int(d.loc[d["RAZON"] > factor, "ID_MUNICIPIO"].nunique()),
        })
    return base, pd.DataFrame(filas)


# ============================================================================
# CONTINUIDAD (aproximación a la oportunidad)
# ============================================================================
def continuidad(df):
    """
    Continuidad del reporte por proveedor.

    El formato establece un plazo de hasta 45 días calendario tras finalizar
    el trimestre, de modo que la oportunidad se define como el cumplimiento
    de ese plazo. La base publicada no registra la fecha de cargue, por lo
    que el plazo no es verificable con la información disponible y el
    indicador mide, en rigor, la continuidad de la serie: se consideran los
    trimestres comprendidos entre el primer y el último reporte de cada
    proveedor, de modo que no se penaliza el ingreso o la salida del mercado
    durante el periodo. La continuidad es condición necesaria pero no
    suficiente de la oportunidad: un proveedor puede reportar los dieciséis
    trimestres y haberlo hecho siempre fuera de plazo.
    """
    trimestres = sorted(df["ORDEN"].unique())
    pos = {o: i for i, o in enumerate(trimestres)}
    n_trim = len(trimestres)

    g = df.groupby("ID_EMPRESA")["ORDEN"]
    rep = g.nunique()
    esperados = g.max().map(pos) - g.min().map(pos) + 1
    vacios = (esperados - rep).clip(lower=0)

    t = pd.DataFrame({
        "Indicador": ["Trimestres del periodo",
                      "Proveedores únicos",
                      "Proveedores con reporte en todos los trimestres",
                      "Proveedores con interrupciones en su serie",
                      "Trimestres esperados (suma)",
                      "Trimestres efectivamente reportados (suma)"],
        "Valor": [n_trim, int(rep.size), int((rep == n_trim).sum()),
                  int((vacios > 0).sum()), int(esperados.sum()), int(rep.sum())]
    })
    indicador = 100 * rep.sum() / esperados.sum()
    return t, indicador


# Nombre conservado por compatibilidad con versiones anteriores del pipeline.
oportunidad = continuidad
