# Cuaderno de Experimentos (Sección 13)

Experimentos controlados sobre el MLP de riesgo crediticio. En todos se mantiene
fija la semilla (42) y el mismo conjunto de validación. La métrica de comparación
es la pérdida de validación (Binary Cross-Entropy) final tras 50 épocas.

Evidencia ejecutable: `notebooks/07_regularization.ipynb` · Datos: `notebooks/reports/experimentos.csv`

| ID | Cambio | Hipótesis | Val Loss (BCE) | Decisión |
| :--- | :--- | :--- | :---: | :--- |
| EXP-01 | Baseline MLP (6-16-8-1) | Referencia inicial | 0.3583 | Seleccionado |
| EXP-07 | Batch size 16 | Lotes más pequeños estabilizan el gradiente | 0.3604 | Descartado (sin mejora) |
| EXP-08 | Early stopping (patience=5) | Detener antes evita sobreajuste | 0.3619 | Descartado (no había overfitting) |
| EXP-02 | Dropout 0.3 | Reduce sobreajuste | 0.3624 | Descartado |
| EXP-06 | Dropout 0.3 + WD 1e-4 | Combinar regularizaciones | 0.3626 | Descartado |
| EXP-04 | Weight decay 1e-4 | Penalización L2 ligera | 0.3684 | Descartado |
| EXP-05 | Weight decay 1e-3 | Penalización L2 fuerte | 0.3697 | Descartado |
| EXP-03 | Dropout 0.5 | Regularización agresiva | 0.3841 | Descartado (peor) |

## Interpretación

- El **modelo base** obtiene la menor pérdida de validación. Las curvas de entrenamiento
  y validación descienden juntas y sin divergencia, lo que indica que **no hay
  sobreajuste** que las técnicas de regularización deban corregir.
- Añadir Dropout o weight decay reduce la capacidad del modelo sin aportar generalización
  en este caso, por lo que incrementan ligeramente la pérdida.
- El Dropout 0.5 es la peor configuración: regulariza en exceso para un modelo ya pequeño.
- **Conclusión:** se conserva la arquitectura base (Sección 6) como configuración final.
  La regularización se documenta como alternativa evaluada y descartada con evidencia.
