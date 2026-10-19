"""
figura_percentiles.py — Genera la figura de percentiles de las variables
derivadas acotadas sin reentrenar el modelo

La figura solo describe la distribución de las cuatro variables derivadas que
se acotan, de modo que no requiere ejecutar el Isolation Forest.

Requiere haber ejecutado antes run_calidad.py.

Uso:
    python figura_percentiles.py
"""

import config as cfg
import informe as inf

df, prom = inf.cargar_o_preparar(cfg)
if cfg.SEGMENTO_MODELO:
    df = df[df["SEG_GRUPO"] == cfg.SEGMENTO_MODELO].reset_index(drop=True)

inf.configurar_estilo()
ruta = inf.fig_percentiles(df, cfg, cfg.DIR_FIG, prom)

print(f"Figura generada sobre {len(df):,} registros del segmento "
      f"{cfg.SEGMENTO_MODELO.lower()}: {ruta}")
