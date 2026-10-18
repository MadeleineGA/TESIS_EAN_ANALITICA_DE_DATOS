"""
informe.py — Carga de datos, generación del informe HTML y figuras
Reporte T.1.3 de servicios fijos · CRC

Todas las figuras se producen en escala de grises, conforme al criterio de
presentación adoptado en el documento.
"""

import base64
import mimetypes
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------------------------- FORMATO DE LAS CIFRAS
# Las cifras se escriben con la convención del español: punto para los miles
# y coma para los decimales. La conversión se aplica sobre las celdas de
# datos del HTML, nunca sobre los encabezados.
# El script arreglar_informe.py reproduce esta misma conversión para los
# informes generados con versiones anteriores; se mantiene allí de forma
# independiente para que pueda ejecutarse sin importar este módulo.
_RE_MILES = re.compile(r"^-?\d{1,3}(?:,\d{3})+$")
_RE_DECIMAL = re.compile(r"^-?\d{1,3}(?:,\d{3})*\.\d+$")
_RE_CIENTIFICA = re.compile(r"^-?\d+(?:\.\d+)?[eE][+-]?\d+$")
_RE_CELDA = re.compile(r"(<td[^>]*>)(.*?)(</td>)", re.S)


def _numero_es(txt):
    t = txt.strip()
    if not t:
        return None
    if _RE_MILES.match(t):
        return t.replace(",", ".")
    if _RE_DECIMAL.match(t):
        ent, _, dec = t.rpartition(".")
        return ent.replace(",", ".") + "," + dec
    if _RE_CIENTIFICA.match(t):
        v = float(t)
        if 0 < abs(v) < 0.001:
            return "&lt; 0,001"
        return t
    return None


# Columnas cuyos enteros son identificadores o periodos y no cantidades: no
# llevan separador de miles. Un periodo 20253 no debe escribirse 20.253.
_COLS_SIN_MILES = re.compile(
    r"\bper[ií]odos?\b|\ba[ñn]os?\b|\banno\b|\borden\b|\bc[oó]digos?\b|"
    r"\bnit\b|^id_|\bsemillas?\b|\btrimestre\b", re.I)


def enteros_con_miles(df):
    """
    Escribe los enteros de las columnas de cantidades con separador de miles.
    Se omiten las columnas de identificadores y periodos, y los valores
    inferiores a mil, que no lo requieren.
    """
    d = df.copy()
    for c in d.columns:
        if _COLS_SIN_MILES.search(str(c)):
            continue
        if pd.api.types.is_bool_dtype(d[c]) or not pd.api.types.is_integer_dtype(d[c]):
            continue
        d[c] = d[c].map(
            lambda v: "" if pd.isna(v) else (f"{v:,}" if abs(v) >= 1000 else str(v)))
    return d


def _formato_decimal(x):
    """
    Dos decimales como mínimo y cuatro como máximo, sin ceros superfluos:
    0,8312 conserva su precisión y 358,80 no se convierte en 358,8.
    """
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ""
    s = f"{x:,.4f}"
    ent, _, dec = s.rpartition(".")
    dec = dec.rstrip("0")
    while len(dec) < 2:
        dec += "0"
    return f"{ent}.{dec}"


def cifras_en_espanol(html):
    """Convierte las cifras de las celdas de datos a la convención del español."""
    def _celda(m):
        nuevo = _numero_es(m.group(2))
        return m.group(0) if nuevo is None else m.group(1) + nuevo + m.group(3)
    return _RE_CELDA.sub(_celda, html)


CSS = """
<style>
body{font-family:Georgia,'Times New Roman',serif;max-width:1100px;margin:40px auto;
     padding:0 24px;color:#1a1a1a;line-height:1.6;}
h1{font-size:1.9em;border-bottom:2px solid #333;padding-bottom:8px;}
h2{font-size:1.4em;margin-top:2.2em;border-bottom:1px solid #bbb;padding-bottom:5px;}
h3{font-size:1.1em;margin-top:1.6em;color:#333;}
table{border-collapse:collapse;margin:14px 0;font-size:0.86em;
      font-family:Helvetica,Arial,sans-serif;}
th{border-top:1.5px solid #333;border-bottom:1px solid #333;padding:7px 12px;
   text-align:left;font-weight:600;background:#f7f7f7;}
td{border-bottom:1px solid #e0e0e0;padding:6px 12px;}
tr:last-child td{border-bottom:1.5px solid #333;}
td:nth-child(n+2){text-align:right;}
.tabtitulo{font-weight:600;margin-top:24px;margin-bottom:2px;}
.nota{font-size:0.82em;color:#555;font-style:italic;margin-top:4px;}
figure{margin:26px 0;text-align:center;}
img{max-width:100%;border:1px solid #ddd;}
figcaption{font-size:0.85em;color:#444;font-style:italic;margin-top:8px;}
.dato{background:#f2f2f2;padding:2px 6px;font-weight:600;}
.pie{font-size:0.8em;color:#666;margin-top:40px;border-top:1px solid #ccc;
     padding-top:10px;}
</style>
"""


class Informe:
    """Acumula el contenido del informe y lo escribe como HTML."""

    def __init__(self, titulo_documento="Resultados de la detección de anomalías",
                 subtitulo=None):
        self.bloques = [f"<html><head><meta charset='utf-8'>"
                        f"<title>{titulo_documento}</title>{CSS}</head><body>"]
        self.titulo(titulo_documento, 1)
        if subtitulo:
            self.bloques.append(f"<p class='nota'>{subtitulo}</p>")
        self.n_tabla = 0
        self.n_figura = 0
        # Posición en self.bloques y ruta de cada figura, para resolver la
        # ruta al guardar: la que se escribe aquí es relativa al directorio
        # de ejecución y no al directorio en que quedará el informe.
        self._figuras = []

    # ---------------------------------------------------------------- texto
    def p(self, texto):
        self.bloques.append(f"<p>{texto}</p>")
        print(texto.replace("<span class='dato'>", "").replace("</span>", ""))

    def titulo(self, texto, nivel=2):
        self.bloques.append(f"<h{nivel}>{texto}</h{nivel}>")
        print(f"\n{'#' * nivel} {texto}")

    # --------------------------------------------------------------- tablas
    def tabla(self, df, titulo=None, nota=None, max_filas=30, mostrar=12,
              numerar=True):
        """
        Inserta una tabla. Si numerar es True, el número se asigna de forma
        automática y consecutiva, de modo que no puedan repetirse.
        """
        if titulo:
            if numerar:
                self.n_tabla += 1
                titulo = f"Tabla {self.n_tabla}. {titulo}"
            self.bloques.append(f'<div class="tabtitulo">{titulo}</div>')
            print(f"\n{titulo}")
        self.bloques.append(cifras_en_espanol(
            enteros_con_miles(df.head(max_filas))
            .to_html(index=False, border=0, float_format=_formato_decimal)))
        if nota:
            self.bloques.append(f'<div class="nota">{nota}</div>')
        print(df.head(mostrar).to_string(index=False))
        return self.n_tabla

    # -------------------------------------------------------------- figuras
    def figura(self, ruta, caption, numerar=True):
        if numerar:
            self.n_figura += 1
            caption = f"Figura {self.n_figura}. {caption}"
        self._figuras.append((len(self.bloques), ruta))
        self.bloques.append(
            f'<figure><img src="{ruta}"><figcaption>{caption}</figcaption></figure>')
        print(f"\n[{caption[:70]}...]")
        return self.n_figura

    def _resolver_figuras(self, destino, incrustar):
        """
        Sustituye la ruta de cada figura por la imagen codificada en base64,
        de modo que el informe sea un único archivo autocontenido. Con
        incrustar=False se escribe en su lugar la ruta relativa al directorio
        en que queda guardado el informe.
        """
        dir_destino = os.path.dirname(os.path.abspath(destino))
        for i, ruta in self._figuras:
            if not os.path.isfile(ruta):
                print(f"  aviso: no se encontró la figura {ruta}")
                continue
            if incrustar:
                tipo = mimetypes.guess_type(ruta)[0] or "image/png"
                with open(ruta, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                nueva = f"data:{tipo};base64,{b64}"
            else:
                nueva = os.path.relpath(os.path.abspath(ruta),
                                        dir_destino).replace(os.sep, "/")
            self.bloques[i] = self.bloques[i].replace(
                f'<img src="{ruta}">', f'<img src="{nueva}">', 1)

    # -------------------------------------------------------------- cierre
    def guardar(self, ruta, pie=None, incrustar_figuras=True):
        carpeta = os.path.dirname(ruta)
        if carpeta:
            os.makedirs(carpeta, exist_ok=True)
        self._resolver_figuras(ruta, incrustar_figuras)
        if pie:
            self.bloques.append(f'<div class="pie">{pie}</div>')
        self.bloques.append("</body></html>")
        with open(ruta, "w", encoding="utf-8") as f:
            f.write("\n".join(self.bloques))
        tam = os.path.getsize(ruta) / 1e6
        print(f"\nInforme generado: {ruta} "
              f"({self.n_tabla} tablas, {self.n_figura} figuras, {tam:.1f} MB)")


def miles(x):
    """Entero con separador de miles en la convención del español."""
    return f"{int(x):,}".replace(",", ".")


def decimales(x, d=2):
    """Número con punto de miles y coma decimal."""
    return (f"{x:,.{d}f}".replace(",", "\x01").replace(".", ",")
            .replace("\x01", "."))


def pie_de_ejecucion(cfg, n_registros=None):
    """Texto con los parámetros con que se produjo el informe."""
    import platform
    import sklearn
    from datetime import datetime
    partes = [
        f"Ejecutado el {datetime.now():%Y-%m-%d %H:%M}",
        f"Python {platform.python_version()}",
        f"scikit-learn {sklearn.__version__}",
        f"pandas {pd.__version__}",
        f"numpy {np.__version__}",
        f"semilla {cfg.SEED}",
        f"árboles {cfg.N_ARBOLES}",
        f"submuestra {cfg.MAX_SAMPLES}",
        f"contaminación {decimales(100 * cfg.CONTAMINACION, 1)}%",
        f"peso local {decimales(cfg.PESO_LOCAL, 1)}",
        f"velocidades {'logarítmicas' if cfg.LOG_VELOCIDADES else 'en escala original'}",
    ]
    if n_registros:
        partes.insert(1, f"{miles(n_registros)} registros")
    return " · ".join(partes)


# ============================================================================
# CARGA Y PREPARACIÓN
# ============================================================================
def cargar(cfg):
    """Lee el archivo, aplica el filtro del periodo y agrega los marcadores."""
    df = pd.read_csv(cfg.ARCHIVO, sep=cfg.SEP, encoding=cfg.ENCODING,
                     dtype=cfg.DTYPES, decimal=cfg.DECIMAL, low_memory=False)
    df.columns = df.columns.str.strip().str.upper()
    n_original = len(df)

    df = df[(df["ANNO"] >= cfg.ANNO_INI) & (df["ANNO"] <= cfg.ANNO_FIN)].copy()
    df["PERIODO"] = df["ANNO"].astype(str) + "-" + df["TRIMESTRE"].astype(str) + "T"
    df["ORDEN"] = df["ANNO"] * 10 + df["TRIMESTRE"]
    df["ES_INTERNET"] = df["ID_SERVICIO_PAQUETE"].isin(cfg.SERVICIOS_INTERNET)
    df["SEG_GRUPO"] = np.where(df["ID_SEGMENTO"].isin(cfg.SEGMENTO_CORPORATIVO),
                               "Corporativo", "Residencial")
    return df.reset_index(drop=True), n_original


def _ruta_alterna(ruta):
    return os.path.splitext(ruta)[0] + ".pkl"


def guardar_cache(df, ruta):
    """
    Guarda la preparación intermedia. Intenta parquet y recurre a pickle si
    el motor no está disponible o la escritura falla, de modo que la cadena
    de ejecución no dependa de una dependencia opcional.
    """
    os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
    try:
        df.to_parquet(ruta, index=False)
        return ruta
    except Exception as e:                                   # noqa: BLE001
        alt = _ruta_alterna(ruta)
        print(f"  parquet no disponible ({type(e).__name__}); se guarda {alt}")
        df.to_pickle(alt)
        return alt


def leer_cache(ruta):
    """Lee la preparación intermedia en cualquiera de los dos formatos."""
    if os.path.exists(ruta):
        try:
            return pd.read_parquet(ruta)
        except Exception as e:                               # noqa: BLE001
            print(f"  no se pudo leer {ruta} ({type(e).__name__})")
    alt = _ruta_alterna(ruta)
    if os.path.exists(alt):
        return pd.read_pickle(alt)
    return None


def cargar_o_preparar(cfg):
    """
    Devuelve (df, prom). Reutiliza la preparación de run_calidad.py si está
    disponible; en caso contrario vuelve a leer el archivo original.
    """
    df = leer_cache(cfg.CACHE_DF)
    prom = leer_cache(cfg.CACHE_PROM)
    if df is not None and prom is not None:
        print(f"Preparación previa reutilizada: {len(df):,} registros.")
        return df, prom
    print("No se encontró la preparación previa; leyendo el archivo original...")
    df, _ = cargar(cfg)
    return df, promedios_municipales(df)


def promedios_municipales(df):
    """
    Velocidad promedio ponderada por accesos, calculada por trimestre,
    municipio y segmento.

    Replica el criterio institucional de cálculo del indicador: la velocidad
    de referencia corresponde al promedio ponderado por el número de accesos,
    de modo que refleja la capacidad efectivamente recibida por los usuarios
    y no el promedio de los planes ofertados. El cálculo por trimestre evita
    que la evolución de las velocidades a lo largo del periodo se interprete
    como comportamiento atípico.
    """
    base = df[df["ES_INTERNET"] & (df["VELOCIDAD_EFECTIVA_DOWNSTREAM"] > 0)].copy()
    base["PESO"] = base["CANTIDAD_LINEAS_ACCESOS"].clip(lower=0).fillna(0)
    base["NUM_DOWN"] = base["VELOCIDAD_EFECTIVA_DOWNSTREAM"] * base["PESO"]
    base["NUM_UP"] = base["VELOCIDAD_EFECTIVA_UPSTREAM"] * base["PESO"]

    claves = ["ANNO", "TRIMESTRE", "ID_MUNICIPIO", "SEG_GRUPO"]

    seg = (base.groupby(claves, observed=True)
               .agg(NUM_DOWN=("NUM_DOWN", "sum"),
                    NUM_UP=("NUM_UP", "sum"),
                    ACCESOS=("PESO", "sum"),
                    N=("PESO", "size"))
               .reset_index())
    seg["VEL_PROMEDIO"] = seg["NUM_DOWN"] / seg["ACCESOS"].replace(0, np.nan)
    seg["VEL_PROMEDIO_UP"] = seg["NUM_UP"] / seg["ACCESOS"].replace(0, np.nan)
    seg = seg.drop(columns=["NUM_DOWN", "NUM_UP"])

    mun = (base.groupby(["ANNO", "TRIMESTRE", "ID_MUNICIPIO"], observed=True)
               .agg(NUM=("NUM_DOWN", "sum"), DEN=("PESO", "sum"))
               .reset_index())
    mun["VEL_PROMEDIO_MUN"] = mun["NUM"] / mun["DEN"].replace(0, np.nan)
    mun = mun.drop(columns=["NUM", "DEN"])

    return seg.merge(mun, on=["ANNO", "TRIMESTRE", "ID_MUNICIPIO"], how="left")


# ============================================================================
# FIGURAS
# ============================================================================
def configurar_estilo():
    plt.rcParams.update({"figure.dpi": 200, "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.3,
                         "axes.prop_cycle": plt.cycler(color=["0.2", "0.45",
                                                              "0.65", "0.8"])})


def _guardar(fig, dir_fig, nombre):
    os.makedirs(dir_fig, exist_ok=True)
    ruta = f"{dir_fig}/{nombre}"
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


def fig_distribucion(df, umbral, contaminacion, dir_fig):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["SCORE_COMBINADO"], bins=120, color="0.45")
    ax.axvline(umbral, color="0.1", ls="--", lw=1.2,
               label=f"Umbral ({contaminacion*100:.0f}%)")
    ax.set_xlabel("Puntuación combinada de anomalía")
    ax.set_ylabel("Registros")
    ax.set_yscale("log")
    ax.legend()
    return _guardar(fig, dir_fig, "fig_distribucion.png")


def fig_cola_puntuacion(df, umbral, dir_fig, columna="SCORE_COMBINADO",
                        pct=0.05):
    """
    Detalle del extremo superior de la distribución de la puntuación.

    Permite verificar si el conjunto señalado constituye un grupo
    desprendido del resto —lo que se apreciaría como un valle entre ambas
    masas— o el extremo de una distribución continua.
    """
    s = df[columna]
    corte = s.quantile(1 - pct)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(s[s >= corte], bins=120, color="0.55")
    ax.axvline(umbral, color="0.1", ls="--", lw=1.4, label="Umbral de corte")
    ax.set_xlabel("Puntuación combinada de anomalía")
    ax.set_ylabel("Registros")
    ax.set_title(f"Extremo superior de la distribución ({pct:.0%} de mayor "
                 f"puntuación)", fontsize=9, loc="left")
    ax.legend()
    return _guardar(fig, dir_fig, "fig_cola_puntuacion.png")


def fig_por_trimestre(df, dir_fig):
    s = (df.groupby("PERIODO")
           .agg(Total=("ANOMALIA", "size"), A=("ANOMALIA", "sum")).reset_index())
    s["Tasa"] = 100 * s["A"] / s["Total"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(s["PERIODO"], s["Tasa"], color="0.45")
    ax.set_ylabel("Anomalías detectadas (%)")
    ax.set_xlabel("Trimestre")
    plt.xticks(rotation=45, ha="right")
    return _guardar(fig, dir_fig, "fig_trimestre.png")


def fig_global_local(df, dir_fig, rng, n=40000):
    m = rng.choice(len(df), min(n, len(df)), replace=False)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(df["SCORE_GLOBAL"].values[m], df["SCORE_LOCAL"].values[m],
               s=2, alpha=0.15, color="0.4")
    ax.set_xlabel("Puntuación global")
    ax.set_ylabel("Puntuación local (por municipio)")
    return _guardar(fig, dir_fig, "fig_global_local.png")


def fig_diffi(importancia, dir_fig, top_n=12):
    top = importancia.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["Variable"], top["Importancia"], color="0.45")
    ax.axvline(1, color="0.1", ls="--", lw=1)
    ax.set_xlabel("Importancia relativa (DIFFI)")
    return _guardar(fig, dir_fig, "fig_diffi.png")


def fig_tipologia(tip, dir_fig):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(tip["Tipo de anomalía"], tip["Registros"], color="0.45")
    for i, v in enumerate(tip["Registros"]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Registros identificados")
    return _guardar(fig, dir_fig, "fig_tipologia.png")


def fig_velocidad_segmento(base_vel, dir_fig):
    from matplotlib.ticker import FixedLocator, FuncFormatter

    fig, ax = plt.subplots(figsize=(7, 4.5))
    datos = [base_vel.loc[base_vel["SEG_GRUPO"] == s, "VELOCIDAD_EFECTIVA_DOWNSTREAM"]
             for s in ["Residencial", "Corporativo"]]
    estilo = dict(showfliers=False, patch_artist=True,
                  boxprops=dict(facecolor="0.85", color="0.3"),
                  medianprops=dict(color="0.1", linewidth=1.6),
                  whiskerprops=dict(color="0.3"), capprops=dict(color="0.3"))
    etiquetas = ["Residencial", "Corporativo"]
    try:                       # matplotlib >= 3.9
        ax.boxplot(datos, tick_labels=etiquetas, **estilo)
    except TypeError:          # versiones anteriores
        ax.boxplot(datos, labels=etiquetas, **estilo)

    ax.set_yscale("log")
    marcas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    ax.yaxis.set_major_locator(FixedLocator(marcas))
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:,.0f}".replace(",", ".") if v >= 1
        else f"{v:.3f}".rstrip("0").replace(".", ",")))

    ax.set_ylabel("Velocidad de descarga (Mbps)")
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", visible=False)
    return _guardar(fig, dir_fig, "fig_velocidad.png")


def fig_transformacion_log(df, dir_fig):
    """Distribución de la cantidad de accesos antes y después del logaritmo."""
    acc = df["CANTIDAD_LINEAS_ACCESOS"].fillna(0).clip(lower=0)
    log_acc = np.log1p(acc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.hist(acc, bins=100, color="0.45")
    ax1.set_yscale("log")
    ax1.set_xlabel("Cantidad de líneas o accesos")
    ax1.set_ylabel("Número de registros")
    ax1.set_title("a) Escala original", fontsize=10, loc="left")

    ax2.hist(log_acc, bins=100, color="0.45")
    ax2.set_xlabel("log(1 + cantidad de líneas o accesos)")
    ax2.set_ylabel("Número de registros")
    ax2.set_title("b) Escala logarítmica", fontsize=10, loc="left")

    return _guardar(fig, dir_fig, "fig_transformacion_log.png")


def fig_percentiles(df, cfg, dir_fig, promedios=None):
    """
    Curva de percentiles de las cuatro variables derivadas que se acotan.

    Justifica los topes definidos en config.py: la curva permanece plana
    hasta el entorno del percentil 99 y se dispara verticalmente en la cola,
    de modo que un número reducido de observaciones determinaría el recorrido
    de la variable y, con él, el particionamiento del algoritmo. Las series se
    calculan sin acotar, para que la figura muestre qué recorta cada tope.
    """
    from matplotlib.ticker import FuncFormatter

    def _limpia(s):
        return (s.replace([np.inf, -np.inf], np.nan).dropna())

    series = []

    v = (df["VALOR_FACTURADO_O_COBRADO"]
         / df["CANTIDAD_LINEAS_ACCESOS"].replace(0, np.nan))
    series.append(("VALOR_POR_ACCESO", "Valor facturado por acceso (COP)",
                   _limpia(v), cfg.TOPE_VALOR_POR_ACCESO))

    r = (df["VELOCIDAD_EFECTIVA_DOWNSTREAM"]
         / df["VELOCIDAD_EFECTIVA_UPSTREAM"].replace(0, np.nan))
    series.append(("RATIO_DOWN_UP", "Razón descarga / carga",
                   _limpia(r), cfg.TOPE_RATIO_DOWN_UP))

    if promedios is not None:
        claves = ["ANNO", "TRIMESTRE", "ID_MUNICIPIO", "SEG_GRUPO"]
        aux = df[claves].merge(promedios[claves + ["VEL_PROMEDIO"]],
                               on=claves, how="left")
        m = (df["VELOCIDAD_EFECTIVA_DOWNSTREAM"].values
             / aux["VEL_PROMEDIO"].replace(0, np.nan).values)
        series.append(("VEL_RELATIVA_MUNICIPIO", "Veces el promedio municipal",
                       _limpia(pd.Series(m)), cfg.TOPE_VEL_REL_MUNICIPIO))

    med = (df[df["ES_INTERNET"]]
           .groupby("ID_TECNOLOGIA")["VELOCIDAD_EFECTIVA_DOWNSTREAM"]
           .median().replace(0, np.nan))
    t = df["VELOCIDAD_EFECTIVA_DOWNSTREAM"] / df["ID_TECNOLOGIA"].map(med)
    series.append(("VEL_RELATIVA_TECNOLOGIA", "Veces la mediana de su tecnología",
                   _limpia(t), cfg.TOPE_VEL_REL_TECNOLOGIA))

    q = np.linspace(0.90, 1.0, 400)
    filas = int(np.ceil(len(series) / 2))
    fig, axes = plt.subplots(filas, 2, figsize=(10, 4.2 * filas))
    axes = np.atleast_1d(axes).ravel()

    for ax, (nombre, etiqueta, s, tope) in zip(axes, series):
        s = s[s > 0]
        ax.plot(100 * q, s.quantile(q).to_numpy(), "-", color="0.25", lw=1.4)
        ax.axhline(tope, color="0.1", ls="--", lw=1.2,
                   label=f"Tope aplicado: {miles(tope)}")
        ax.set_yscale("log")
        ax.set_xlabel("Percentil de la distribución")
        ax.set_ylabel(etiqueta)
        ax.set_title(nombre, fontsize=9, loc="left")
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: miles(v) if v >= 1 else f"{v:g}".replace(".", ",")))
        ax.legend(fontsize=7.5, loc="upper left")

        n_ac = int((s > tope).sum())
        ax.annotate(f"Acotados: {miles(n_ac)} ({decimales(100*n_ac/len(s), 4)} %)",
                    xy=(0.97, 0.06), xycoords="axes fraction", ha="right",
                    fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec="0.6", lw=0.6))

    for ax in axes[len(series):]:
        ax.set_visible(False)

    return _guardar(fig, dir_fig, "fig_percentiles.png")


def fig_percentiles_velocidades(df, dir_fig):
    """
    Recorrido y concentración de la dispersión de las velocidades efectivas.

    La fila superior presenta la curva de percentiles en escala logarítmica:
    el recorrido se despliega a lo largo de varios órdenes de magnitud y
    asciende de forma abrupta en el último percentil. La fila inferior
    descompone la varianza de la variable según el tramo de la distribución
    al que pertenece cada registro, y muestra que la dispersión la aporta un
    número muy reducido de observaciones. Ambas se calculan sobre los
    servicios que incluyen internet fijo con velocidad mayor que cero, que es
    la población sobre la que la variable tiene sentido.

    Sustenta la decisión de conservar las velocidades en su escala original:
    los registros que determinan el recorrido de la variable son los que el
    modelo debe aislar, de modo que comprimir esa cola equivaldría a atenuar
    la señal que se busca.
    """
    from matplotlib.ticker import FuncFormatter

    base = df[df["ES_INTERNET"]] if "ES_INTERNET" in df.columns else df

    series = [
        ("VELOCIDAD_EFECTIVA_DOWNSTREAM", "Velocidad de descarga (Mbps)"),
        ("VELOCIDAD_EFECTIVA_UPSTREAM", "Velocidad de carga (Mbps)"),
    ]

    q = np.concatenate([np.linspace(0.0, 0.99, 300),
                        np.linspace(0.99, 1.0, 400)])
    etiquetas = ["Hasta el\npercentil 90", "Percentil\n90 a 99",
                 "Percentil\n99 a 99,9", "Sobre el\npercentil 99,9"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for j, (col, etiqueta) in enumerate(series):
        s = (base[col].astype("float64")
             .replace([np.inf, -np.inf], np.nan).dropna())
        s = s[s > 0]
        p50, p90, p99, p999 = s.quantile([0.50, 0.90, 0.99, 0.999])

        # --- Fila superior: recorrido de la variable ---
        ax = axes[0, j]
        ax.plot(100 * q, s.quantile(q).to_numpy(), "-", color="0.25", lw=1.4)
        ax.axvline(99, color="0.55", ls=":", lw=1.0)
        ax.set_yscale("log")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percentil de la distribución")
        ax.set_ylabel(etiqueta)
        ax.set_title(col, fontsize=9, loc="left")
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: miles(v) if v >= 1 else f"{v:g}".replace(".", ",")))
        ax.annotate(f"Mediana: {miles(p50)}\n"
                    f"Percentil 99: {miles(p99)}\n"
                    f"Máximo: {miles(s.max())}",
                    xy=(0.04, 0.96), xycoords="axes fraction",
                    ha="left", va="top", fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec="0.6", lw=0.6))

        # --- Fila inferior: aporte de cada tramo a la varianza ---
        ax = axes[1, j]
        desv = (s - s.mean()) ** 2
        limites = [s.min() - 1, p90, p99, p999, s.max()]
        tramo = pd.cut(s, bins=limites, labels=etiquetas, include_lowest=True)
        aporte = (100 * desv.groupby(tramo, observed=False).sum() / desv.sum())
        aporte = aporte.reindex(etiquetas).fillna(0)
        n_tramo = (s.groupby(tramo, observed=False).size()
                   .reindex(etiquetas).fillna(0))

        barras = ax.bar(range(len(etiquetas)), aporte.to_numpy(),
                        color="0.45", width=0.62)
        ax.set_xticks(range(len(etiquetas)))
        ax.set_xticklabels(etiquetas, fontsize=7.5)
        ax.set_ylim(0, 112)
        ax.set_ylabel("Aporte a la varianza (%)")
        ax.grid(axis="x", visible=False)

        for b, v, n in zip(barras, aporte.to_numpy(), n_tramo.to_numpy()):
            ax.annotate(f"{decimales(v, 1)} %\n{miles(int(n))} reg.",
                        xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5)

    return _guardar(fig, dir_fig, "fig_percentiles_velocidades.png")


def fig_camino_score(detector, X, umbral, dir_fig, rng, n=50000):
    """
    Relación entre la longitud promedio de camino y la puntuación de anomalía.

    Verifica empíricamente el comportamiento descrito por la formulación del
    algoritmo: los registros que se aíslan con menos particiones reciben una
    puntuación más alta.
    """
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    Xs = X.values[idx]

    # Profundidad de la hoja en que termina cada observación, promediada
    prof = np.zeros(len(Xs))
    for est in detector.modelo_global.estimators_:
        t = est.tree_
        d = np.zeros(t.node_count, dtype=np.int32)
        pila = [(0, 0)]
        while pila:
            nodo, p = pila.pop()
            d[nodo] = p
            if t.children_left[nodo] != -1:
                pila.append((t.children_left[nodo], p + 1))
                pila.append((t.children_right[nodo], p + 1))
        prof += d[est.apply(Xs)]
    prof /= len(detector.modelo_global.estimators_)

    score = -detector.modelo_global.score_samples(Xs)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(prof, score, s=2, alpha=0.15, color="0.4")
    ax.axhline(umbral, color="0.1", ls="--", lw=1.2, label="Umbral de corte")
    ax.set_xlabel("Longitud promedio de camino  E[h(x)]")
    ax.set_ylabel("Puntuación de anomalía  s(x, n)")
    ax.legend()
    return _guardar(fig, dir_fig, "fig_camino_score.png")


def fig_curva_magnitud(tabla, dir_fig,
                       x="Factor de alteración",
                       cols=("ROC-AUC", "Exhaustividad en el umbral del 1% (%)")):
    """Desempeño en función de la magnitud del error inyectado."""
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(tabla[x], tabla[cols[0]], "o-", color="0.2", lw=1.4,
             label=cols[0])
    ax1.set_xscale("log")
    ax1.set_xlabel("Factor de alteración (escala logarítmica)")
    ax1.set_ylabel(cols[0])
    ax1.set_ylim(0, 1.02)

    ax2 = ax1.twinx()
    ax2.plot(tabla[x], tabla[cols[1]], "s--", color="0.55", lw=1.4,
             label=cols[1])
    ax2.set_ylabel("Exhaustividad en el umbral del 1% (%)")
    ax2.set_ylim(0, 102)
    ax2.grid(False)

    lineas = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lineas, [l.get_label() for l in lineas], fontsize=8,
               loc="lower right")
    return _guardar(fig, dir_fig, "fig_curva_magnitud.png")


def fig_comparacion_algoritmos(tabla, dir_fig, orden=None):
    """ROC-AUC de cada algoritmo en cada escenario de inyección.

    orden : secuencia con los nombres de los escenarios en el orden en que
        deben representarse. Si es None se conserva el orden alfabético que
        produce la tabla dinámica. El eje horizontal se rotula con el número
        de orden de cada escenario y la correspondencia entre número y
        nombre se declara en el pie de la figura.
    """
    piv = tabla.pivot(index="Escenario", columns="Algoritmo", values="ROC-AUC")
    if orden is not None:
        piv = piv.reindex([e for e in orden if e in piv.index])
    tramas = ["", "///", "...", "xxx"]
    x = np.arange(len(piv))
    ancho = 0.8 / len(piv.columns)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (col, tr) in enumerate(zip(piv.columns, tramas)):
        ax.bar(x + i * ancho, piv[col], ancho, label=col,
               color="0.75", edgecolor="0.2", hatch=tr)
    ax.set_xticks(x + 0.4 - ancho / 2)
    ax.set_xticklabels([str(i + 1) for i in range(len(piv))], fontsize=9)
    ax.set_xlabel("Escenario de inyección")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="0.1", ls=":", lw=1)
    ax.legend(fontsize=8, ncol=2)
    return _guardar(fig, dir_fig, "fig_comparacion_algoritmos.png")
