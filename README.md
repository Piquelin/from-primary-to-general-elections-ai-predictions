# from-primary-to-general-elections-ai-predictions
Creo este repostorio para usar como soporte en el taller de la media party 2023

La idea es ver si se puede entrenar un modelo con los datos de las paso-generales de 2015 y 2019 que tienen la ventaja de ser presidenciales y a nivel nacional con las mismas listas.
Y con ese modelo entrenao hacer una predicción sobre el resultado de la elecciones del 2023 contando como entrada las paso 2023.

Metodológicamente estamos pasando por alto varias diferencias históricas y de contexto que diferenciasn las 3 elecciones presidenciales que estamos analizando. Hacemos esto concientes de que hay muchas cosas que afectan los resultados electorales y que además son distíntas para cada elección. Pero de todas formas nos parecía util de una manera exploratoria y didáctica. Tanto para analizar las diferentes elecciones como para familiarizarnos en el pipeline de entrenar un modelo de aprendizaje automático.

El programa "paso_a_generales_distritos.py" toma los datos de las consultas hechas a la base de datos de La Izquierda Diario (https://observatorio.laizquierdadiario.com/consultas) y los agrupa por circuitos similares usando kmeans.

La carpeta "datos_circuitos" contiene bajadas de los resultados de las paso y generales junto con el nomenclador de las agrupaciones. Como material base para trabajar si no contamos  con la capacidad de conectar a la base directo. 

