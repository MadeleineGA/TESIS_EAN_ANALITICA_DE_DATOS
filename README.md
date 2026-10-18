# Detección de anomalías en el reporte de servicios fijos de la CRC

Código que produce los resultados de la monografía **«Modelo de aprendizaje
automático no supervisado para la detección de anomalías en la información
reportada de servicios fijos ante la Comisión de Regulación de Comunicaciones
de Colombia»** (Maestría en Ciencia de Datos, Universidad EAN).

El repositorio permite reproducir íntegramente las tablas y figuras que el
documento reporta.
---

## 1. Qué hace

A partir del reporte **T.1.3 «Líneas o accesos y valores facturados o cobrados
de servicios fijos individuales y empaquetados»**, publicado por la CRC para el
periodo 2022–2025, el código ejecuta tres procesos independientes:

| Proceso | Script | Salida |
|---|---|---|
| Evaluación de las dimensiones de calidad a partir de las reglas de validación del formato | `run_calidad.py` | `informe_calidad.html` |
| Modelo de detección de anomalías (Isolation Forest global y segmentado por municipio, atribución DIFFI, tipología y métricas) | `run_modelo.py` | `informe_modelo.html`, `anomalias_detectadas.csv` |
| Validación mediante inyección controlada de anomalías y comparación con algoritmos alternativos | `run_validacion.py` | `informe_validacion.html` |

Cada informe HTML contiene las tablas con sus notas y las figuras, numeradas de
forma automática y consecutiva por el módulo `informe.py`, de modo que la
numeración de las salidas no pueda duplicarse.

---

## 2. Datos de entrada

El archivo de entrada **no se incluye en el repositorio** por su tamaño
(aproximadamente 3,4 millones de registros). Es información pública y se
descarga del portal de datos abiertos de la CRC (Postdata):

 https://www.postdata.gov.co/dataset/empaquetamiento-de-servicios-fijos

- Formato: T.1.3 — Líneas o accesos y valores facturados o cobrados de
  servicios fijos individuales y empaquetados.
- Periodo utilizado: 2022-1T a 2025-4T.
- El archivo debe guardarse como `entradas/EMPAQUETAMIENTO_FIJO.csv`.

Características del archivo que el código asume (definidas en `config.py`):
separador `;`, codificación `utf-8-sig`, separador decimal `,`.

Columnas esperadas: `ANNO`, `TRIMESTRE`, `ID_EMPRESA`, `EMPRESA`,
`ID_MUNICIPIO`, `MUNICIPIO`, `ID_DEPARTAMENTO`, `DEPARTAMENTO`, `ID_SEGMENTO`,
`SEGMENTO`, `ID_SERVICIO_PAQUETE`, `SERVICIO_PAQUETE`, `ID_TECNOLOGIA`,
`TECNOLOGIA`, `ID_ESTADO`, `ESTADO`, `VELOCIDAD_EFECTIVA_DOWNSTREAM`,
`VELOCIDAD_EFECTIVA_UPSTREAM`, `CANTIDAD_LINEAS_ACCESOS`,
`VALOR_FACTURADO_O_COBRADO`, `OTROS_VALORES_FACTURADOS`.

---

## 3. Instalación y ejecución

### Disposición de las carpetas

Las rutas definidas en `config.py` suponen que el código reside en una carpeta
llamada `codigo_final` y que los scripts se ejecutan **desde el directorio que
la contiene**, en el que se ubica también la carpeta `entradas`:

```
proyecto/
├── entradas/
│   └── EMPAQUETAMIENTO_FIJO.csv
└── codigo_final/
    ├── config.py, informe.py, calidad.py, …
    ├── figuras/      (se crea en la primera ejecución)
    └── salidas/      (se crea en la primera ejecución)
```

```bash
git clone https://github.com/<usuario>/crc-anomalias-servicios-fijos.git codigo_final

python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r codigo_final/requirements.txt

mkdir -p entradas                  # colocar aquí EMPAQUETAMIENTO_FIJO.csv

python codigo_final/run_calidad.py     # debe ejecutarse primero
python codigo_final/run_modelo.py
python codigo_final/run_validacion.py
```

Para ejecutar desde dentro de la propia carpeta basta con retirar el prefijo
`codigo_final/` de las rutas del bloque de salida de `config.py`. Todas las
rutas están centralizadas allí y ninguna aparece escrita en los demás módulos.

`run_calidad.py` deja preparada en `salidas/` la base con los marcadores
derivados y la tabla de promedios municipales, que los dos scripts siguientes
reutilizan. Si esos archivos no existen, cada script vuelve a leer el CSV
original.

### Regeneración de figuras sin reentrenar

Dos figuras describen únicamente la distribución de las variables y no
requieren ejecutar el modelo, de modo que pueden regenerarse en menos de un
minuto una vez ejecutado `run_calidad.py`:

```bash
python codigo_final/figura_percentiles.py   # variables derivadas acotadas
python codigo_final/figura_velocidades.py   # velocidades reportadas
```

### Tiempos de ejecución aproximados

Medidos sobre un equipo con 8 núcleos y 32 GB de memoria, con el archivo
completo:

| Script | Tiempo | Paso más costoso |
|---|---|---|
| `run_calidad.py` | 5–10 min | Lectura del CSV |
| `run_modelo.py` | 2,5–4 h | Modelos locales por municipio (≈940 bosques) y comparación de criterios de segmentación |
| `run_validacion.py` | 40–70 min | Comparación de algoritmos sobre cinco escenarios |

El paso más lento de `run_modelo.py` es el apartado 7.3, que reentrena todos
los modelos locales de cada criterio de segmentación. Puede desactivarse
comentando ese bloque sin afectar al resto del informe.

### Informes autocontenidos y formato de las cifras

Cada informe incrusta sus figuras codificadas en base64, de modo que queda
como un único archivo que puede enviarse o archivarse sin la carpeta
`figuras/`. Las cifras se escriben con la convención del español —punto para
los miles, coma para los decimales— y los valores p inferiores a 0,001 se
expresan en esa forma convencional.

---

## 4. Estructura del repositorio

```
├── config.py              Parámetros. Todo valor que condiciona un resultado está aquí
├── informe.py             Carga del archivo, clase Informe (HTML) y figuras
├── calidad.py             Dimensiones de calidad a partir de las reglas del formato
├── modelo.py              Variables de entrada, Isolation Forest global y local, DIFFI, tipología
├── metricas.py            Indicadores de evaluación calculables sin etiquetas
├── comparacion.py         Comparación de decisiones de diseño
├── validacion.py          Inyección controlada de anomalías y algoritmos alternativos
├── run_calidad.py         Informe de calidad
├── run_modelo.py          Informe del modelo
├── run_validacion.py      Informe de validación
├── figura_percentiles.py  Regenera la figura de las variables derivadas acotadas
├── figura_velocidades.py  Regenera la figura de las velocidades reportadas
├── requirements.txt       Dependencias
├── entradas/              Archivo de entrada (no versionado)
├── salidas/               Informes HTML, CSV de anomalías y preparación intermedia
└── figuras/               Figuras en PNG, escala de grises
```

---

## 5. Parámetros y origen de cada valor

Todos están en `config.py`. La distinción importa porque solo uno procede de
los autores del algoritmo:

| Parámetro | Valor | Origen |
|---|---|---|
| `MAX_SAMPLES` (ψ) | 256 | Valor de referencia propuesto por Liu, Ting y Zhou (2008) |
| `N_ARBOLES` (t) | 400 | Decisión de este trabajo. Los autores proponen 100. El valor se fijó a partir del análisis de convergencia que produce `metricas.convergencia_arboles`, que contrasta cada tamaño de bosque con uno de referencia superior al adoptado |
| `CONTAMINACION` | 1 % |  Criterio operativo asociado a la capacidad de revisión de la entidad. No interviene en la construcción del modelo: los árboles, las puntuaciones y el ordenamiento resultante son idénticos con cualquier valor. Determina únicamente en qué percentil se corta ese ordenamiento y, por tanto, cuántos registros integran el conjunto señalado. Quien replique el ejercicio puede modificarlo según su propia capacidad de revisión; obtendrá las mismas puntuaciones y un conjunto señalado de distinto tamaño, anidado respecto del original. Su efecto se documenta en la tabla de sensibilidad al umbral.|
| `PESO_LOCAL` | 0,5 | Decisión de este trabajo. No se deriva de una optimización, porque sobre los datos reales no existe una medida de desempeño que maximizar. El efecto de la decisión se acota con el núcleo común entre ponderaciones |
| `MIN_POR_SEGMENTO` | 200 | Decisión de este trabajo. Número mínimo de observaciones para que un bosque local tenga capacidad discriminante |
| `SEGMENTO_LOCAL` | `ID_MUNICIPIO` | Adaptación del Layered Isolation Forest (T. Liu et al., 2024): los subespacios se definen por una variable del reporte y no por partición geométrica |
| `LOG_VELOCIDADES` | `False` | Decisión de este trabajo. Los accesos y los dos campos de valores facturados se incorporan como log(1+v); las dos velocidades se conservan en su escala original. El algoritmo no supone normalidad ni simetría en las variables de entrada, de modo que la transformación altera la detección únicamente por la probabilidad con que cada corte aísla una observación. Cambiar el valor y volver a ejecutar `run_modelo.py` reproduce la especificación alternativa |
| `TOPE_*` | ver `config.py` | Decisión de este trabajo, a partir de la curva de percentiles de cada variable. Se aplican únicamente a las variables derivadas de un cociente y no a las velocidades reportadas |
| `UMBRAL_TEMPORAL`, `BASE_TEMPORAL` | 300 %, 50 accesos | Decisión de este trabajo, a partir de la exploración de los datos |
| `FACTOR_COLECTIVA`, `MIN_COLECTIVA` | 3, 5 | Decisión de este trabajo. La categoría colectiva se define como tasa de señalamiento del grupo proveedor–trimestre, no como conteo absoluto |
| `PCT_MULTIVAR` | 0,25 | Decisión de este trabajo. El umbral multivariado se calibra con el percentil 25 de la distribución observada del peso de la variable dominante |
| `SEED` | 42 | Reproducibilidad |

---

## 6. Reproducibilidad

- Toda la aleatoriedad está controlada por `config.SEED`. Con la misma
  semilla, el mismo archivo de entrada y las mismas versiones de las
  librerías, los resultados son idénticos.
- Cada informe HTML cierra con un pie que registra la fecha de ejecución, las
  versiones de Python, scikit-learn, pandas y numpy, y los parámetros con que
  se produjo, incluida la escala aplicada a las velocidades. Ese pie es el que
  permite comprobar que un informe corresponde a una configuración determinada.
- El resultado **no** es idéntico entre versiones distintas de scikit-learn:
  la implementación del Isolation Forest ha cambiado entre versiones. Las
  versiones con que se produjeron los resultados del documento están fijadas
  en `requirements.txt`.
- La identificación de registros no es única: ejecuciones con distintas
  semillas producen conjuntos señalados que no coinciden por completo. El
  informe del modelo incluye las tablas que miden esa coincidencia y que
  localizan los registros que varían dentro de la distribución.

### Librerías utilizadas

`pandas` y `numpy` para la manipulación de datos; `scikit-learn` para el
Isolation Forest, el Local Outlier Factor y el análisis de componentes
principales; `scipy` para la correlación de Spearman y la prueba
hipergeométrica; `matplotlib` para las figuras. El método DIFFI y el
algoritmo HBOS se implementaron a partir de su definición: el primero sobre la
estructura de árboles de scikit-learn y el segundo sobre histogramas de
anchura variable. No se utiliza `seaborn`.

---

## 7. Alcance y limitaciones del código

- **No hay registros etiquetados.** Las medidas de desempeño supervisado que
  reporta `run_validacion.py` se calculan sobre anomalías inyectadas de forma
  controlada, no sobre errores confirmados. Establecen la capacidad del modelo
  para recuperar errores del tipo que se le introduce, bajo el supuesto de que
  la base de partida está libre de errores de esa misma naturaleza.
- **El contraste con las reglas del formato no mide acierto.** Las reglas son
  un referente imperfecto: un registro puede satisfacerlas y ser incorrecto.
  Lo que el contraste establece es complementariedad, mediante la comparación
  del solapamiento observado con el esperado bajo independencia.
- **El alcance se limita al segmento residencial.** La provisión al segmento
  corporativo responde a requerimientos particulares de cada suscriptor, lo
  que introduce una dispersión que el modelo interpretaría como atípica. Su
  tratamiento requeriría una estrategia de segmentación distinta.
- **El componente contextual evalúa cada registro frente a su municipio.** Un
  proveedor cuya oferta difiera de la predominante en el territorio será
  señalado por construcción, con independencia de que el valor reportado sea
  correcto. El informe del modelo incluye la comparación de criterios de
  segmentación que incorpora la tecnología de acceso como dimensión
  adicional.
- **La continuidad no es la oportunidad.** La base publicada no registra la
  fecha de cargue, de modo que el plazo de 45 días no es verificable con la
  información disponible. El indicador mide continuidad de la serie, que es
  condición necesaria pero no suficiente de la oportunidad.

---

## 8. Licencia y uso

El código se publica bajo licencia MIT (ver `LICENSE`). Los datos de entrada
son información pública publicada por la CRC y se rigen por las condiciones de
uso de su portal de datos abiertos.

Este repositorio es un trabajo académico de carácter individual. No constituye
un desarrollo institucional de la Comisión de Regulación de Comunicaciones ni
compromete su posición.

### Cómo citar

> Gil Ángel, M. (2026). *Modelo de aprendizaje automático no supervisado para
> la detección de anomalías en la información reportada de servicios fijos ante
> la Comisión de Regulación de Comunicaciones de Colombia* [Monografía de
> maestría]. Universidad EAN.

### Uso de herramientas de inteligencia artificial

Se utilizó una herramienta de inteligencia artificial como apoyo en la
revisión de estilo y la depuración del código. El diseño metodológico, el
análisis, la interpretación y las conclusiones son responsabilidad de la
autora.
