"""
run_validacion.py — Validación del modelo mediante inyección controlada de
anomalías y comparación con algoritmos alternativos

Genera: salidas/informe_validacion.html

Produce las medidas de desempeño supervisado que el reporte no permite
calcular por sí mismo —ROC-AUC, precisión promedio y exhaustividad en el
umbral de operación— sobre casos de error conocido, y contrasta el
Isolation Forest con tres algoritmos alternativos sobre los mismos
conjuntos.

Requiere haber ejecutado antes run_calidad.py.

Uso:
    python run_validacion.py
"""

import numpy as np
import pandas as pd

import config as cfg
import informe as inf
import validacion as val

rep = inf.Informe(
    "Validación del modelo mediante inyección controlada de anomalías",
    subtitulo="Reporte T.1.3 · Comisión de Regulación de Comunicaciones · "
              f"periodo {cfg.ANNO_INI}–{cfg.ANNO_FIN}")

inf.configurar_estilo()

# ============================================================================
# 0. BASE DE PARTIDA
# ============================================================================
rep.titulo("1. Procedimiento")


df, prom = inf.cargar_o_preparar(cfg)
if cfg.SEGMENTO_MODELO:
    df = df[df["SEG_GRUPO"] == cfg.SEGMENTO_MODELO].reset_index(drop=True)

base = val.base_limpia(df, n=cfg.N_BASE_VALIDACION, seed=cfg.SEED)

rep.tabla(pd.DataFrame({
    "Elemento del diseño": [
        "Registros del segmento analizado",
        "Registros que satisfacen todas las reglas y tienen velocidad y accesos positivos",
        "Base de referencia empleada (submuestra aleatoria)",
        "Anomalías inyectadas por escenario",
        "Proporción de contaminación inyectada (%)",
        "Umbral de operación del modelo (%)",
        "Semilla aleatoria"],
    "Valor": [f"{len(df):,}",
              f"{int((~val.incumple_reglas(df) & df['ES_INTERNET'] & (df['VELOCIDAD_EFECTIVA_DOWNSTREAM'] > 0) & (df['CANTIDAD_LINEAS_ACCESOS'] > 0)).sum()):,}",
              f"{len(base):,}",
              f"{cfg.N_INYECCION:,}",
              round(100 * cfg.N_INYECCION / len(base), 3),
              round(100 * cfg.CONTAMINACION, 2),
              cfg.SEED]}),
    "Diseño del experimento de validación",
    "Nota. La proporción de anomalías inyectadas (0,50%) se fija por debajo "
    "del umbral de operación (1,00%) para que el conjunto señalado pueda "
    "contenerlas en su totalidad. De este modo la proporción recuperada no "
    "queda acotada por el diseño del experimento y refleja únicamente la capacidad "
    "de detección del modelo.")

MAG = val.MAGNITUDES_ESCENARIO

def _mil(n):
    return f"{n:,}".replace(",", ".")

rep.tabla(pd.DataFrame({
    "Escenario": [val.NOMBRES_ESCENARIO[e] for e in val.ESCENARIOS],
    "Alteración introducida": [
        f"Las dos velocidades del registro se multiplican por un factor de "
        f"{_mil(MAG['unidad'])}, lo que convierte el reporte de Mbps a kbps.",
        "Se intercambian los valores de descarga y de carga en los registros "
        "en los que la primera supera a la segunda al menos al doble.",
        f"El valor facturado se multiplica por un factor de "
        f"{_mil(MAG['facturacion'])}, lo que genera un valor por acceso "
        "desproporcionado.",
        f"El número de accesos se multiplica por un factor de "
        f"{_mil(MAG['accesos'])}, lo que reproduce un error de magnitud en el "
        "campo.",
        f"Se asigna una velocidad igual al promedio del municipio multiplicado "
        f"por un factor de {_mil(MAG['contextual'])}, en municipios cuyo "
        "promedio se sitúa por debajo del percentil 40."],
    "Correspondencia con los hallazgos de las dimensiones de calidad, la "
    "construcción de las variables o la clasificación de anomalías": [
        "Registros con velocidades superiores a 100.000 Mbps.",
        "Registros con velocidad de carga superior a la de descarga.",
        "Registros con valores facturados de cientos de millones sobre un "
        "único acceso.",
        "Variaciones abruptas en el número de accesos entre trimestres.",
        "Registros con una velocidad muy superior al promedio de su "
        "municipio."]}),
    "Escenarios para la inyección de anomalías",
    "Nota. Los escenarios no son hipotéticos: cada uno reproduce un patrón de "
    "diligenciamiento efectivamente observado en el reporte y documentado en "
    "el informe de calidad. Los factores de alteración corresponden a las "
    "magnitudes de los casos observados en cada patrón.")


# ============================================================================
# 2. DESEMPEÑO POR ESCENARIO
# ============================================================================
rep.titulo("2. Desempeño del modelo por escenario")

t_esc = val.evaluar_escenarios(base, prom, cfg)
rep.tabla(t_esc, "Desempeño del Isolation Forest por escenario de inyección",
          "Nota. La precisión en el umbral del 1% está acotada por el diseño: "
          "el conjunto señalado tiene un tamaño fijo, de modo que no puede "
          "exceder la proporción entre anomalías inyectadas y registros "
          "señalados. La columna de exhaustividad es la que informa sobre la "
          "capacidad de recuperación.")


# ============================================================================
# 3. CURVA DE SENSIBILIDAD
# ============================================================================
rep.titulo("3. Sensibilidad a la magnitud del error")

t_curva = val.curva_magnitud(base, prom, cfg, escenario="unidad")
rep.tabla(t_curva, "Desempeño según la magnitud de la alteración (confusión en "
                   "la unidad de medida)",
          "Nota. El factor expresa el múltiplo aplicado a la velocidad "
          "reportada. El factor 1.000 corresponde a la confusión entre Mbps y "
          "kbps observada en el reporte.")

rep.figura(inf.fig_curva_magnitud(t_curva, cfg.DIR_FIG),
           "Área bajo la curva ROC y exhaustividad en el umbral de operación "
           "según la magnitud del error inyectado. El eje horizontal está en "
           "escala logarítmica.")


# ============================================================================
# 4. COMPARACIÓN CON ALGORITMOS ALTERNATIVOS
# ============================================================================
rep.titulo("4. Comparación con algoritmos alternativos")

rep.tabla(pd.DataFrame({
    "Algoritmo": ["Isolation Forest", "HBOS", "LOF (submuestra)",
                  "PCA (error de reconstrucción)"],
    "Principio": [
        "Aislamiento mediante particiones aleatorias del espacio",
        "Densidad univariada estimada por histogramas, agregada entre variables",
        "Densidad local relativa a los vecinos más próximos",
        "Distancia al subespacio que concentra la varianza del conjunto"],
    "Limitación esperada en este contexto": [
        "Particiones paralelas a los ejes; desempeño documentado como inferior "
        "ante anomalías locales",
        "Supone independencia entre variables, por lo que no detecta anomalías "
        "definidas por la relación entre campos",
        "Costo computacional cuadrático; se evalúa sobre submuestra",
        "Capta desviaciones de la estructura lineal; no identifica el campo "
        "responsable"]}),
    "Algoritmos comparados",
    "Nota. HBOS se implementó a partir de su definición, en la variante de "
    "anchura variable definida por cuantiles, adecuada ante distribuciones "
    "muy asimétricas. Los tres restantes provienen de scikit-learn.")

t_alg, r_alg = val.comparar_algoritmos(base, prom, cfg)
rep.tabla(r_alg, "Desempeño medio de cada algoritmo entre escenarios",
          "Nota. El promedio entre escenarios no debe leerse como una medida "
          "única de calidad: un algoritmo puede obtener un promedio alto y "
          "fallar por completo en un escenario determinado. La columna de "
          "ROC-AUC mínimo informa sobre ese comportamiento.")

rep.tabla(t_alg, "Desempeño de cada algoritmo en cada escenario", max_filas=40)

rep.figura(
    inf.fig_comparacion_algoritmos(
        t_alg, cfg.DIR_FIG,
        orden=[val.NOMBRES_ESCENARIO[e] for e in val.ESCENARIOS]),
    "Área bajo la curva ROC de cada algoritmo en cada escenario de "
    "inyección. La línea horizontal señala el valor correspondiente a una "
    "asignación aleatoria. Los escenarios del eje horizontal son: "
    + "; ".join(f"{i + 1}, {val.NOMBRES_ESCENARIO[e][0].lower()}"
                f"{val.NOMBRES_ESCENARIO[e][1:]}"
                for i, e in enumerate(val.ESCENARIOS))
    + ".")


# ============================================================================
# 5. APORTE DE LA SEGMENTACIÓN
# ============================================================================
rep.titulo("5. Aporte de la segmentación por municipio")

t_segm, n_mod, cob = val.contraste_segmentacion(base, prom, cfg)
rep.tabla(t_segm, "Desempeño de cada configuración sobre anomalías contextuales",
          f"Nota. Calculado sobre la base de referencia con "
          f"{cfg.N_INYECCION:,} anomalías contextuales inyectadas. Se "
          f"entrenaron {n_mod:,} modelos locales, con una cobertura del "
          f"{cob:.1f}% de los registros.")


# ============================================================================
# 6. GUARDADO
# ============================================================================
rep.guardar(cfg.SALIDA_VALID, pie=inf.pie_de_ejecucion(cfg, len(base)))
