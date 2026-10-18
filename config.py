"""
config.py — Parámetros del pipeline de detección de anomalías
en el reporte T.1.3 de servicios fijos · CRC

Todos los valores que condicionan el resultado están centralizados en este
archivo. Cada bloque indica el origen del valor adoptado: referencia de los
autores del algoritmo, regla del formato, decisión del presente trabajo
verificada empíricamente, o convención operativa.
"""

# ------------------------------------------------------------------ ENTRADA
ARCHIVO  = "entradas/EMPAQUETAMIENTO_FIJO.csv"
SEP      = ";"
ENCODING = "utf-8-sig"
DECIMAL  = ","

ANNO_INI, ANNO_FIN = 2022, 2025

# ------------------------------------------------------- REGLAS DEL FORMATO
# Servicios que incluyen internet fijo: deben reportar tecnología y velocidad.
# Origen: diccionario del formato T.1.3.
SERVICIOS_INTERNET = [1, 4, 5, 7]

# Segmentos no residenciales: corporativo (107), uso propio interno del
# operador (109) y accesos adicionales corporativos (117). El resto de los
# códigos del catálogo corresponde al segmento residencial por estrato.
SEGMENTO_CORPORATIVO = [107, 109, 117]

# --------------------------------------------------------- MODELO: GLOBAL
# CONTAMINACION: criterio operativo del presente trabajo, asociado a la
#   capacidad de revisión de la entidad. No interviene en el entrenamiento:
#   opera únicamente al convertir la puntuación continua en decisión binaria.
#   Su efecto se documenta en la tabla de sensibilidad al umbral.
# MAX_SAMPLES: valor de referencia propuesto por Liu, Ting y Zhou (2008).
# N_ARBOLES: decisión del presente trabajo a partir del análisis de
#   convergencia (metricas.convergencia_arboles). Los autores proponen 100;
#   la coincidencia entre conjuntos señalados seguía aumentando en ese valor,
#   por lo que se adopta 400.
CONTAMINACION = 0.01
N_ARBOLES     = 400
MAX_SAMPLES   = 256
SEED          = 42

# Segmento sobre el que opera el modelo
SEGMENTO_MODELO = "Residencial"   # "Residencial", "Corporativo" o None (ambos)

# ------------------------------------------- MODELO: SEGMENTADO (Layered IF)
SEGMENTO_LOCAL   = ["ID_MUNICIPIO"]  # variables que definen los subespacios
MIN_POR_SEGMENTO = 200               # mínimo de registros para modelo local
PESO_LOCAL       = 0.5               # ponderación entre global y local

# ------------------------------------------------------------------- DIFFI
MAX_DIFFI = 8000   # tope de registros para el cálculo de atribución global
BLOQUE_DIFFI = 5000  # tamaño del bloque en la atribución local

# ------------------------------------------ TOPES DE LAS VARIABLES DERIVADAS
# Acotan los valores extremos para evitar que un número reducido de
# observaciones domine el particionamiento del algoritmo. El acotamiento
# preserva la condición extrema del registro: conserva el valor máximo de la
# variable y continúa siendo señalado.
TOPE_VALOR_POR_ACCESO   = 1e7    # pesos facturados por acceso
TOPE_RATIO_DOWN_UP      = 1000   # razón entre velocidad de descarga y carga
TOPE_VEL_REL_MUNICIPIO  = 100    # veces el promedio de su municipio
TOPE_VEL_REL_TECNOLOGIA = 100    # veces la mediana de su tecnología

# Transformación de las velocidades absolutas. El valor True las incorpora
# como log(1+v), igual que los accesos y los valores facturados; el valor
# False las deja en su escala original.
#
# Se adopta False. El Isolation Forest no supone normalidad ni simetría en
# las variables de entrada, de modo que corregir la asimetría no constituye
# por sí mismo una razón para transformarlas. La dispersión de ambas
# velocidades proviene de un número muy reducido de registros del extremo de
# la distribución —entre ellos los errores de unidad de medida— que son
# precisamente los que el modelo debe aislar, y el contraste empírico de
# ambas especificaciones favoreció la escala original en los indicadores de
# estabilidad del conjunto señalado y de separación. Cambiar este valor y
# volver a ejecutar run_modelo.py reproduce la especificación alternativa.
LOG_VELOCIDADES = False

# -------------------------------------------------- UMBRALES DE TIPOLOGÍA
UMBRAL_TEMPORAL   = 300   # variación porcentual entre trimestres consecutivos
BASE_TEMPORAL     = 50    # accesos mínimos del periodo previo

# Colectiva: se define como tasa y no como conteo absoluto, de modo que la
# categoría no dependa del tamaño del proveedor. Un grupo proveedor–trimestre
# es colectivo si su tasa de señalamiento supera FACTOR_COLECTIVA veces la
# tasa general del modelo y reúne al menos MIN_COLECTIVA registros señalados.
FACTOR_COLECTIVA = 3
MIN_COLECTIVA    = 5

# Multivariada: se calibra en tiempo de ejecución con el percentil
# PCT_MULTIVAR de la distribución del peso de la variable dominante.
PCT_MULTIVAR    = 0.25
UMBRAL_MULTIVAR = 0.30   # valor por defecto si no se recalibra

# ---------------------------------------------------- EVALUACIÓN DEL MODELO
SEMILLAS_ESTABILIDAD = (42, 101, 202, 303, 404)
ARBOLES_CONVERGENCIA = (25, 50, 100, 200, 400, 800)
PESOS_COMPARACION    = (0.0, 0.25, 0.5, 0.75, 1.0)
UMBRALES_COMPARACION = (0.005, 0.01, 0.02, 0.05)

# ------------------------------------------- VALIDACIÓN POR INYECCIÓN
# Número de anomalías sintéticas por escenario y magnitudes evaluadas en la
# curva de sensibilidad. La inyección permite calcular medidas de desempeño
# supervisado sobre casos de error conocido.
N_INYECCION      = 2000
MAGNITUDES       = (2, 5, 10, 50, 100, 1000)
N_BASE_VALIDACION = 400000   # tamaño de la base limpia de referencia

# ------------------------------------------------------------------ SALIDA
DIR_FIG         = "codigo_final/figuras"
DIR_SALIDA      = "codigo_final/salidas"
SALIDA_CALIDAD  = "codigo_final/salidas/informe_calidad.html"
SALIDA_MODELO   = "codigo_final/salidas/informe_modelo.html"
SALIDA_VALID    = "codigo_final/salidas/informe_validacion.html"
SALIDA_CSV      = "codigo_final/salidas/anomalias_detectadas.csv"
CACHE_DF        = "codigo_final/salidas/datos_preparados.parquet"
CACHE_PROM      = "codigo_final/salidas/promedios_municipales.parquet"

# ----------------------------------------------------- TIPOS DE LAS COLUMNAS
DTYPES = {
    "ANNO": "int16", "TRIMESTRE": "int8",
    "ID_EMPRESA": "int64", "ID_MUNICIPIO": "float32",
    "ID_DEPARTAMENTO": "float32", "ID_SEGMENTO": "float32",
    "ID_SERVICIO_PAQUETE": "float32", "ID_TECNOLOGIA": "float32",
    "ID_ESTADO": "float32",
    "VELOCIDAD_EFECTIVA_DOWNSTREAM": "float32",
    "VELOCIDAD_EFECTIVA_UPSTREAM": "float32",
    "CANTIDAD_LINEAS_ACCESOS": "float64",
    "VALOR_FACTURADO_O_COBRADO": "float64",
    "OTROS_VALORES_FACTURADOS": "float64",
    "EMPRESA": "category", "MUNICIPIO": "category",
    "DEPARTAMENTO": "category", "SEGMENTO": "category",
    "SERVICIO_PAQUETE": "category", "TECNOLOGIA": "category",
    "ESTADO": "category",
}
