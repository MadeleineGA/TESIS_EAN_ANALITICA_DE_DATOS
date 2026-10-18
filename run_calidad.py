"""
run_calidad.py — Informe de las dimensiones de calidad del reporte de
servicios fijos

Genera: salidas/informe_calidad.html

Evalúa completitud, exactitud, consistencia y continuidad del reporte a
partir de las reglas de validación de primer nivel del formato, y añade dos
verificaciones de plausibilidad que no forman parte de esas reglas.

La numeración de las tablas y figuras del informe es automática y
consecutiva (clase Informe), de modo que no puede repetirse.

Uso:
    python run_calidad.py
"""

import pandas as pd

import config as cfg
import calidad
import informe as inf

rep = inf.Informe(
    "Evaluación de las dimensiones de calidad del reporte de servicios fijos",
    subtitulo="Reporte T.1.3 · Comisión de Regulación de Comunicaciones · "
              f"periodo {cfg.ANNO_INI}–{cfg.ANNO_FIN}")


# ============================================================================
# 1. DATOS ANALIZADOS
# ============================================================================
rep.titulo("1. Datos analizados")

df, n_original = inf.cargar(cfg)

n_res = int((df["SEG_GRUPO"] == "Residencial").sum())
n_cor = int((df["SEG_GRUPO"] == "Corporativo").sum())
emp_res = set(df.loc[df["SEG_GRUPO"] == "Residencial", "ID_EMPRESA"])
emp_cor = set(df.loc[df["SEG_GRUPO"] == "Corporativo", "ID_EMPRESA"])
prov_tot = df["ID_EMPRESA"].nunique()



reg = (df.pivot_table(index=["ORDEN", "PERIODO"], columns="SEG_GRUPO",
                      values="ANNO", aggfunc="size", fill_value=0)
         .rename(columns={"Residencial": "Reg. residencial",
                          "Corporativo": "Reg. corporativo"}))
acc = (df.pivot_table(index=["ORDEN", "PERIODO"], columns="SEG_GRUPO",
                      values="CANTIDAD_LINEAS_ACCESOS", aggfunc="sum", fill_value=0)
         .rename(columns={"Residencial": "Acc. residencial",
                          "Corporativo": "Acc. corporativo"}))
otros = (df.groupby(["ORDEN", "PERIODO"])
           .agg(**{"Proveedores únicos": ("ID_EMPRESA", "nunique"),
                   "Municipios": ("ID_MUNICIPIO", "nunique")}))

vol = (reg.join(acc).join(otros).reset_index()
         .sort_values("ORDEN").drop(columns="ORDEN")
         .rename(columns={"PERIODO": "Periodo"}))
vol["Registros"] = vol["Reg. residencial"] + vol["Reg. corporativo"]
vol = vol[["Periodo", "Reg. residencial", "Reg. corporativo", "Registros",
           "Acc. residencial", "Acc. corporativo", "Proveedores únicos", "Municipios"]]
# Los accesos son recuentos: se presentan como enteros y no con decimales.
for c in ["Acc. residencial", "Acc. corporativo"]:
    vol[c] = vol[c].round().astype("int64")
vol.columns.name = None

rep.tabla(vol, "Volumen de registros y accesos por trimestre y segmento",
          "Nota. Elaboración propia. Los proveedores únicos corresponden al número "
          "de NIT distintos que reportaron en cada trimestre; dado que un mismo "
          "proveedor puede reportar en ambos segmentos, esta cifra no es la suma "
          "de los proveedores de cada uno.", max_filas=20)

prov_seg = pd.DataFrame({
    "Alcance": ["Solo segmento residencial", "Solo segmento corporativo",
                "Reportan en ambos segmentos", "Total de proveedores únicos"],
    "Proveedores": [len(emp_res) - len(emp_res & emp_cor),
                    len(emp_cor) - len(emp_res & emp_cor),
                    len(emp_res & emp_cor), prov_tot]})
prov_seg["% del total"] = (100 * prov_seg["Proveedores"] / prov_tot).round(2)
rep.tabla(prov_seg, "Proveedores según el segmento en que reportan",
          f"Nota. Cifras correspondientes a la totalidad del periodo "
          f"{cfg.ANNO_INI}–{cfg.ANNO_FIN}.")


# ============================================================================
# 2. COMPLETITUD
# ============================================================================
rep.titulo("2. Completitud")


t_comp, ind_completitud = calidad.completitud(df)
rep.tabla(t_comp, "Completitud por variable",
          f"Nota. Indicador global de completitud: {ind_completitud:.2f}%. Se "
          f"evalúan los catorce campos definidos en el diccionario del formato. "
          f"El indicador se calcula sobre la última fila —registros distintos "
          f"con al menos un campo sin informar— y no sobre el promedio de las "
          f"columnas, de modo que sea homogéneo con el de las demás "
          f"dimensiones y no dependa del número de campos evaluados.")

# ============================================================================
# 3. EXACTITUD
# ============================================================================
rep.titulo("3. Exactitud")


t_exa, ind_exactitud, ind_exactitud_total = calidad.exactitud(df)



rep.tabla(t_exa, "Valores fuera de las listas de referencia establecidas para el reporte",
          f"Nota. El indicador se calcula a partir de las dos últimas filas de la tabla, "
          f"que recogen los registros distintos que incumplen al menos una "
          f"verificación. Esa cifra no coincide con la suma de las filas "
          f"anteriores: un mismo registro puede incumplir varias "
          f"verificaciones y se contabiliza una sola vez. Indicador con la "
          f"exclusión de la convención: {ind_exactitud:.2f}%; sin "
          f"exclusiones: {ind_exactitud_total:.2f}%.")

rep.tabla(calidad.tecnologia_en_cero(df),
          "Registros con tecnología reportada en cero, por servicio y segmento",
          "Nota. En los servicios que incluyen internet fijo —códigos 1, 4, 5 y 7— "
          "el valor esperado corresponde a una tecnología de acceso del listado "
          "del diccionario; en los restantes, al código 999.")

rep.tabla(calidad.exactitud_velocidades(df),
          "Rangos de velocidad de descarga por segmento",
          "Nota. Se consideran únicamente los servicios que incluyen internet "
          "fijo con velocidad superior a 0. Los percentiles se calculan "
          "en cada segmento.")


# ============================================================================
# 4. CONSISTENCIA
# ============================================================================
rep.titulo("4. Consistencia")

t_con, ind_consistencia, ind_consistencia_total = calidad.consistencia(df)

rep.tabla(t_con, "Coherencia entre campos",
          f"Nota. Los indicadores se calculan a partir de las dos últimas filas de la "
          f"tabla, que recogen los registros distintos que incumplen al menos "
          f"una regla. Esa cifra no coincide con la suma de las filas "
          f"anteriores: un mismo registro puede incumplir varias reglas y se "
          f"contabiliza una sola vez. Es el caso de los registros que declaran "
          f"velocidad de descarga en cero, que incumplen también la condición de "
          f"que la velocidad de carga no supere a la de descarga. Las reglas se "
          f"derivan del diccionario del formato —los servicios 2, 3 y 6 deben "
          f"reportar velocidad cero y tecnología 999; el segmento 117 solo "
          f"admite los servicios 1, 2 y 3 y exige facturación en cero—, con una "
          f"excepción: la condición de que la velocidad de carga no supere a la "
          f"de descarga se incorpora en este trabajo, dado que el formato no "
          f"establece relación entre ambos campos. "
          f"Indicador con la exclusión de la convención: "
          f"{ind_consistencia:.2f}%; sin exclusiones: "
          f"{ind_consistencia_total:.2f}%.")

# ============================================================================
# 5. CONTINUIDAD DEL REPORTE
# ============================================================================
rep.titulo("5. Continuidad del reporte (aproximación a la oportunidad)")

t_opo, ind_continuidad = calidad.continuidad(df)
rep.tabla(t_opo, "Continuidad del reporte por operador",
          f"Nota. Indicador de continuidad: {ind_continuidad:.2f}%. No se penaliza "
          f"el ingreso ni la salida del mercado durante el periodo.")


# ============================================================================
# 6. SÍNTESIS DE LAS DIMENSIONES
# ============================================================================
rep.titulo("6. Síntesis de las dimensiones evaluadas")

rep.tabla(pd.DataFrame({
    "Dimensión": ["Completitud",
                  "Exactitud (con exclusión de la convención)",
                  "Exactitud (verificación estricta)",
                  "Consistencia (con exclusión de la convención)",
                  "Consistencia (verificación estricta)",
                  "Continuidad del reporte"],
    "Indicador (%)": [round(ind_completitud, 2), round(ind_exactitud, 2),
                      round(ind_exactitud_total, 2),
                      round(ind_consistencia, 2),
                      round(ind_consistencia_total, 2),
                      round(ind_continuidad, 2)],
    "Base de cálculo": ["Registros"] * 5 + ["Trimestres"]}),
    "Síntesis de las dimensiones de calidad",
    "Nota. Los indicadores de completitud, exactitud y consistencia expresan el "
          "porcentaje de registros que satisface la totalidad de las verificaciones "
          "de su dimensión; se calculan sobre registros distintos, de modo que un "
          "registro que incumple varias verificaciones se contabiliza una sola vez. "
          "La exactitud y la consistencia se presentan en dos versiones según el "
          "tratamiento que se dé al código cero en el campo de tecnología. El "
          "indicador de continuidad tiene una base de cálculo distinta: es el "
          "cociente entre los trimestres efectivamente reportados y los esperados "
          "entre el primer y el último reporte de cada proveedor, y no una "
          "proporción de registros, por lo que no resulta directamente comparable "
          "con los tres anteriores.")

# ============================================================================
# 7. PLAUSIBILIDAD DE LA VELOCIDAD EN CONTEXTO TERRITORIAL
# ============================================================================
rep.titulo("7. Verificaciones de plausibilidad ajenas a las reglas del formato")

rep.titulo("7.1 Velocidad frente al promedio de su municipio", 3)

prom = inf.promedios_municipales(df)
base_vel, t_velprom = calidad.velocidad_vs_promedio(df, prom, factor=3)
rep.tabla(t_velprom, "Velocidad reportada frente al promedio municipal",
          "Nota. La razón se calcula como la velocidad del registro dividida "
          "entre la velocidad promedio ponderada de su municipio, segmento y "
          "trimestre.")

rep.tabla(base_vel[base_vel["RAZON"] > 3].nlargest(15, "RAZON")
          [["PERIODO", "EMPRESA", "MUNICIPIO", "SEG_GRUPO", "TECNOLOGIA",
            "VELOCIDAD_EFECTIVA_DOWNSTREAM", "VEL_PROMEDIO", "RAZON"]]
          .rename(columns={"SEG_GRUPO": "Segmento",
                           "VELOCIDAD_EFECTIVA_DOWNSTREAM": "Velocidad",
                           "VEL_PROMEDIO": "Promedio municipal", "RAZON": "Razón"}),
          "Casos con mayor desviación del promedio municipal", max_filas=15)


# ============================================================================
# 7.2 PLAUSIBILIDAD TÉCNICA DE LAS VELOCIDADES
# ============================================================================
rep.titulo("7.2 Plausibilidad técnica de la velocidad declarada", 3)

UMBRAL_EXTREMO = 100000  # Mbps
UMBRAL_TXT = f"{UMBRAL_EXTREMO:,}".replace(",", ".")

inter = df[df["ES_INTERNET"] & (df["VELOCIDAD_EFECTIVA_DOWNSTREAM"] > 0)].copy()

# ---- Distribución por tecnología ----
dist_tec = (inter.groupby("TECNOLOGIA", observed=True)["VELOCIDAD_EFECTIVA_DOWNSTREAM"]
            .agg(Registros="size", Mínima="min", Mediana="median",
                 P95=lambda s: s.quantile(0.95),
                 P999=lambda s: s.quantile(0.999), Máxima="max")
            .round(2).reset_index()
            .sort_values("Registros", ascending=False))
dist_tec = dist_tec[dist_tec["Registros"] > 0]
dist_tec = dist_tec.rename(columns={"TECNOLOGIA": "Tecnología", "P999": "P99.9"})

rep.tabla(dist_tec, "Distribución de la velocidad de descarga por tecnología",
          "Nota. Valores en Mbps. Se consideran únicamente los servicios que "
          "incluyen internet fijo con velocidad mayor que cero.", max_filas=25)

# ---- Registros por encima del umbral extremo ----
ext = inter[inter["VELOCIDAD_EFECTIVA_DOWNSTREAM"] > UMBRAL_EXTREMO]


ext_tec = (ext.groupby("TECNOLOGIA", observed=True)
             .agg(Registros=("ANNO", "size"),
                  Proveedores=("ID_EMPRESA", "nunique"),
                  Máxima=("VELOCIDAD_EFECTIVA_DOWNSTREAM", "max"))
             .reset_index())
ext_tec = ext_tec[ext_tec["Registros"] > 0].sort_values("Registros", ascending=False)
ext_tec["Máxima"] = ext_tec["Máxima"].round(0)
ext_tec = ext_tec.rename(columns={"TECNOLOGIA": "Tecnología",
                                  "Máxima": "Velocidad máxima (Mbps)"})

rep.tabla(ext_tec, f"Registros con velocidad superior a {UMBRAL_TXT} Mbps, "
                   f"según tecnología")

# ---- Concentración por proveedor y segmento ----
ext_emp = (ext.groupby(["EMPRESA", "SEG_GRUPO"], observed=True)
             .agg(Registros=("ANNO", "size"),
                  Municipios=("ID_MUNICIPIO", "nunique"),
                  Máxima=("VELOCIDAD_EFECTIVA_DOWNSTREAM", "max"))
             .reset_index().sort_values("Registros", ascending=False))
ext_emp["Máxima"] = ext_emp["Máxima"].round(0)
ext_emp["% de los casos"] = (100 * ext_emp["Registros"] / len(ext)).round(2)
ext_emp = ext_emp.rename(columns={"SEG_GRUPO": "Segmento",
                                  "Máxima": "Velocidad máxima (Mbps)"})

rep.tabla(ext_emp, "Concentración de los casos extremos por proveedor", max_filas=20)

# ---- Casos individuales ----
cols_ext = [c for c in ["PERIODO", "EMPRESA", "MUNICIPIO", "SEG_GRUPO",
                        "SERVICIO_PAQUETE", "TECNOLOGIA",
                        "VELOCIDAD_EFECTIVA_DOWNSTREAM",
                        "VELOCIDAD_EFECTIVA_UPSTREAM",
                        "CANTIDAD_LINEAS_ACCESOS"] if c in ext.columns]

rep.tabla(ext.nlargest(20, "VELOCIDAD_EFECTIVA_DOWNSTREAM")[cols_ext]
             .rename(columns={"SEG_GRUPO": "Segmento",
                              "VELOCIDAD_EFECTIVA_DOWNSTREAM": "Descarga (Mbps)",
                              "VELOCIDAD_EFECTIVA_UPSTREAM": "Carga (Mbps)",
                              "CANTIDAD_LINEAS_ACCESOS": "Accesos"}),
          "Registros con las velocidades más altas del periodo",
          "Nota. Los valores se expresan en Mbps.",
          max_filas=20)

# ---- Velocidades muy bajas ----
bajas = inter[inter["VELOCIDAD_EFECTIVA_DOWNSTREAM"] < 1]


bajas_tec = (bajas.groupby(["TECNOLOGIA", "SEG_GRUPO"], observed=True)
               .agg(Registros=("ANNO", "size"),
                    Proveedores=("ID_EMPRESA", "nunique"),
                    Mínima=("VELOCIDAD_EFECTIVA_DOWNSTREAM", "min"))
               .reset_index())
bajas_tec = bajas_tec[bajas_tec["Registros"] > 0].sort_values("Registros", ascending=False)
bajas_tec = bajas_tec.rename(columns={"TECNOLOGIA": "Tecnología",
                                      "SEG_GRUPO": "Segmento",
                                      "Mínima": "Velocidad mínima (Mbps)"})

rep.tabla(bajas_tec, "Registros con velocidad inferior a 1 Mbps", max_filas=20)



# ============================================================================
# 7.3 SIMETRÍA ENTRE LA VELOCIDAD DE DESCARGA Y LA DE CARGA
# ============================================================================
SEG_SIM = cfg.SEGMENTO_MODELO          # segmento sobre el que se analiza
TXT_SEG = f" del segmento {SEG_SIM.lower()}" if SEG_SIM else ""

rep.titulo("7.3 Igualdad entre la velocidad de descarga y la de carga", 3)

t_sim, t_sim_tec, t_sim_prov = calidad.simetria_velocidades(df, segmento=SEG_SIM)

rep.tabla(t_sim, "Registros con velocidades de descarga y carga idénticas",
          f"Nota. Se consideran los registros{TXT_SEG} que declaran un servicio "
          f"con internet fijo y ambas velocidades mayores que cero. La "
          f"clasificación de las tecnologías según su capacidad de entregar "
          f"velocidades simétricas está definida en calidad.py y puede "
          f"consultarse allí.")

rep.tabla(t_sim_tec, "Igualdad entre ambas velocidades, por tecnología de acceso",
          f"Nota. Calculada sobre los registros{TXT_SEG} con internet fijo y "
          f"ambas velocidades mayores que cero. La última columna recoge si la "
          f"arquitectura de acceso permite entregar velocidades simétricas: se "
          f"consideran viables los accesos que llevan fibra hasta el "
          f"domicilio, el predio o el edificio, y no viables los que conservan "
          f"un tramo final de cobre o de radio. La categoría de otras "
          f"tecnologías de fibra se clasifica como indeterminada por agrupar "
          f"accesos de arquitectura diversa. "
          f"La simetría se plantea por parte del investigador. No tiene validación "
          f"bibliográfica.", max_filas=25)

rep.tabla(t_sim_prov, "Igualdad entre ambas velocidades en accesos xDSL, por "
                      "proveedor",
          f"Nota. Se incluyen los proveedores con al menos 100 registros"
          f"{TXT_SEG} en esa tecnología. La clasificación de las tecnologías "
          f"según su capacidad de entregar velocidades simétricas es propia de "
          f"este trabajo y no cuenta con respaldo bibliográfico.", max_filas=20)

# ============================================================================
# 8. FIGURAS
# ============================================================================
rep.titulo("8. Figuras")
inf.configurar_estilo()
rep.figura(inf.fig_velocidad_segmento(base_vel, cfg.DIR_FIG),
           "Distribución de la velocidad de descarga por segmento. "
           "Se excluyen los valores extremos para facilitar la lectura.")
rep.figura(inf.fig_percentiles(df, cfg, cfg.DIR_FIG, prom),
           "Curva de percentiles de las cuatro variables derivadas que se acotan en la construcción de la matriz de entrada del modelo, calculada sobre los valores sin acotar.")


# ============================================================================
# 9. GUARDADO
# ============================================================================
rep.guardar(cfg.SALIDA_CALIDAD, pie=inf.pie_de_ejecucion(cfg, len(df)))

# Se conserva la preparación para reutilizarla en los informes siguientes
r1 = inf.guardar_cache(df, cfg.CACHE_DF)
r2 = inf.guardar_cache(prom, cfg.CACHE_PROM)
print(f"Datos preparados guardados: {r1} y {r2}")
