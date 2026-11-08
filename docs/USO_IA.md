# Registro de Uso de IA como Copiloto (Sección 19)

La IA se usó para apoyar la revisión de código, la generación de documentación y el
diseño de experimentos. Todos los resultados fueron verificados por el equipo ejecutando
los notebooks y revisando las métricas.

| Fase | Prompt (plantilla) | Uso de la IA | Validación humana |
| :--- | :--- | :--- | :--- |
| Entrenamiento | "Revisa este código de entrenamiento: busca errores de dimensiones, loss, leakage, reproducibilidad y guardado." | Detectó que el entrenamiento usaba datos sin escalar (loss estancado). | Se corrigió y se verificó el descenso de la pérdida ejecutando `06`. |
| Regularización | "Analiza estas curvas: describe sobreajuste/subajuste y plantea experimentos." | Propuso comparar base vs Dropout/L2/early stopping. | Se ejecutaron y se confirmó que no había sobreajuste. |
| Experimentos | "Crea ocho experimentos controlados para el proyecto." | Diseñó EXP-01 a EXP-08 con semilla y validación fijas. | Se ejecutaron en `07` y se registró `notebooks/reports/experimentos.csv`. |
| Evaluación | "Interpreta resultados: distingue observaciones de hipótesis." | Sugirió métricas y comparación con baseline lineal. | Se calcularon en `08` sobre el Test real. |
| Interpretación | "Diseña una estrategia de interpretación para la red." | Propuso permutation importance (compatible con el MLP). | Se ejecutó en `09`; resultados coherentes con el negocio. |
| Despliegue | "Diseña el flujo de despliegue con preprocesamiento consistente y pruebas." | Definió el flujo de inferencia y los casos de prueba. | Se validó en `10` con 4 casos. |
| Documentación | "Diseña la estructura de repositorio y la documentación." | Redacción de README y documentos de `docs/`. | Revisado y ajustado por el equipo. |

## Controles aplicados

- [x] Verificar las respuestas ejecutando el código.
- [x] No subir datos sensibles ni credenciales.
- [x] Probar el código antes de documentar resultados.
- [x] Registrar los prompts relevantes (esta tabla).
- [x] Citar las fuentes externas (dataset, framework).
- [x] Mantener la responsabilidad humana sobre las decisiones técnicas.
