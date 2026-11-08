# Informe Técnico — Predicción de Riesgo Crediticio con MLP

## Resumen

Se diseñó, entrenó y evaluó un Perceptrón Multicapa (MLP) para predecir la probabilidad
de mora crediticia a 90 días a partir de 6 variables financieras. El modelo alcanza un
ROC-AUC de 0.851 en el conjunto de Test y supera a un baseline de Regresión Logística en
Recall y F1, respondiendo afirmativamente a la pregunta de investigación.

## Introducción

Las entidades financieras necesitan clasificar solicitudes de crédito minimizando tanto
la cartera vencida como el rechazo de clientes solventes. Los enfoques lineales no capturan
las interacciones no lineales entre ingreso, tasa y endeudamiento. Se propone un MLP como
alternativa capaz de aprender fronteras de decisión más complejas.

## Fundamentos

El modelo implementa una neurona artificial con combinación lineal `z = Wx + b` y activación
no lineal. Se usan activaciones ReLU en las capas ocultas y Sigmoid en la salida para obtener
una probabilidad. La función de pérdida es Binary Cross-Entropy y el aprendizaje se realiza
por backpropagation con el optimizador Adam. (Detalle en Secciones 4 a 10 del README.)

## Metodología

- **Datos:** Credit Risk (Kaggle), 32.581 registros. Limpieza de outliers de edad, imputación
  de la tasa por mediana, split 80/20 estratificado y escalamiento MinMax sin data leakage.
- **Arquitectura:** MLP 6 → 16 → 8 → 1 (ReLU + Sigmoid).
- **Entrenamiento:** BCE + Adam (lr=0.001), 50 épocas, batch 32, semilla 42, validación interna.
- **Evaluación:** métricas sobre Test reservado y comparación con Regresión Logística.

## Resultados

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Regresión Logística | 0.831 | 0.698 | 0.398 | 0.507 | 0.827 |
| **MLP** | **0.843** | 0.671 | **0.548** | **0.603** | **0.851** |

- Las curvas de entrenamiento y validación descienden juntas (sin sobreajuste).
- La regularización no aportó mejoras; se conservó el modelo base.
- Variables más influyentes: `loan_percent_income` y `loan_int_rate`.

## Discusión

El MLP mejora la detección de mora frente al enfoque lineal (Recall +15 pts). Su punto débil
es el Recall absoluto (54.8%): aún deja pasar casi la mitad de los morosos, consecuencia del
desbalance de clases. La interpretación confirma que el modelo se apoya en variables
financieramente razonables.

## Conclusiones

El MLP responde a la pregunta de investigación: clasifica la mora con mayor precisión que
la regresión lineal convencional. El proyecto es reproducible (semilla fija, datos y código
versionados) y cuenta con un prototipo de inferencia.

## Trabajo futuro

- Abordar el desbalance (class weights, sobremuestreo) para mejorar el Recall.
- Incorporar más variables del dataset original.
- Ajuste de hiperparámetros más amplio.

## Referencias

- Dataset: Credit Risk (Kaggle, CC0).
- PyTorch — framework de aprendizaje profundo.
- Kingma & Ba (2014), *Adam: A Method for Stochastic Optimization*.
