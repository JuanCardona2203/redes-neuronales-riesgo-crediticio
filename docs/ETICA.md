# Ética, Riesgos y Uso Responsable (Sección 20)

El modelo clasifica solicitudes de crédito, una decisión que afecta directamente a
personas. Por ello se documentan los riesgos y sus mitigaciones.

| Riesgo | ¿Aplica? | Evidencia | Mitigación | Verificación |
| :--- | :---: | :--- | :--- | :--- |
| **Privacidad** | Sí | El dataset contiene datos financieros personales. | Uso de dataset público anonimizado (CC0); no se almacenan datos reales. | No se publican datos personales en el repositorio. |
| **Sesgo** | Sí | Clases desbalanceadas (~21.8% de mora). | Split estratificado; evaluación con Recall/F1, no solo Accuracy. | Métricas por clase en `08_evaluation.ipynb`. |
| **Transparencia** | Sí | Decisiones automáticas de crédito. | Interpretación de variables (permutation importance). | `09_interpretation.ipynb`. |
| **Discriminación** | Sí | Variables como edad podrían introducir sesgo. | La edad resultó casi irrelevante para el modelo. | Importancia de variables en `09`. |
| **Supervisión humana** | Sí | El modelo no debe decidir solo. | La salida es una recomendación (APROBAR/DENEGAR), no una decisión final. | Flujo de inferencia en `10_deployment.ipynb`. |
| **Seguridad** | Sí | Entradas inválidas o maliciosas. | Validación de variables completas y sin nulos antes de inferir. | Casos de prueba en `10`. |
| **Uso indebido** | Sí | Aplicar el modelo fuera de su contexto. | Se documenta que el modelo es un prototipo académico, no apto para producción real. | Limitaciones en el README. |

## Principios

- El modelo es una **herramienta de apoyo**, no un sustituto del criterio humano.
- Un falso negativo (moroso aprobado) implica pérdida de capital; un falso positivo
  (cliente solvente rechazado) implica exclusión financiera injustificada. Ambos errores
  tienen consecuencias y deben revisarse por un analista.
- El modelo no demuestra causalidad: identifica correlaciones con la mora, no sus causas.
