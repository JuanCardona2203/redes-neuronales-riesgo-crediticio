# Aplicación de Despliegue (app)

Demo interactiva del modelo de riesgo crediticio construida con **Streamlit**.

- `app.py`: interfaz que recibe los datos de una solicitud, aplica el mismo escalamiento del entrenamiento y muestra la probabilidad de mora con la decisión sugerida (APROBAR / DENEGAR).

## Ejecución

Requisitos: haber entrenado el modelo (`notebooks/06_training.ipynb`) para generar `models/mlp_credit_risk.pt`.

```bash
streamlit run app/app.py
```
