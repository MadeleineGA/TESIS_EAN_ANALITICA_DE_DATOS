"""
figura_velocidades.py — Genera la figura de percentiles de las velocidades
sin reentrenar el modelo

La figura describe únicamente la distribución de las dos velocidades
reportadas, de modo que no requiere ejecutar el Isolation Forest. Permite
regenerarla, sobre el mismo segmento que emplea el modelo, en menos de un
minuto.

Requiere haber ejecutado antes run_calidad.py.

Uso:
    python figura_velocidades.py
"""

import config as cfg
import informe as inf

df, _ = inf.cargar_o_preparar(cfg)
if cfg.SEGMENTO_MODELO:
    df = df[df["SEG_GRUPO"] == cfg.SEGMENTO_MODELO].reset_index(drop=True)

inf.configurar_estilo()
ruta = inf.fig_percentiles_velocidades(df, cfg.DIR_FIG)

print(f"Figura generada sobre {len(df):,} registros del segmento "
      f"{cfg.SEGMENTO_MODELO.lower()}: {ruta}")
