# Predicción de Riesgo Crediticio mediante Redes Neuronales (MLP)

**Presentado por:** 
---

Yenifer Salazar Restrepo

Juan Esteban Cardona

Camilo Andrés Estrada Sosa

---

## 1. Descripción del Proyecto
Este proyecto implementa un modelo de Inteligencia Artificial basado en una Red Neuronal Densa (Perceptrón Multicapa - MLP) para la clasificación probabilística del riesgo de morosidad a más de 90 días en solicitudes de crédito de consumo bancario.

---

## 2. Definición del Problema y Justificación

### Contexto del Problema
El departamento de riesgos bancarios requiere automatizar la evaluación crediticia. Los modelos lineales tradicionales presentan limitaciones al capturar interacciones no lineales entre los ingresos del cliente, su índice de endeudamiento y su comportamiento histórico de pagos.

| Elemento | Registro del Proyecto |
| :--- | :--- |
| **Contexto** | Evaluación de solicitudes de crédito de consumo. |
| **Problema observable** | Incremento en la tasa de morosidad (>90 días) por clasificaciones imprecisas. |
| **A quién impacta** | Área de gestión de riesgo financiero y clientes solicitantes. |
| **Decisión a mejorar** | Clasificación de aprobación vs. rechazo de créditos. |
| **Oportunidad con RN** | Aprendizaje automático de fronteras de decisión no lineales complejas sin asumir distribuciones previas. |

### Justificación Técnica
* **Reglas de Negocio:** Rígidas e incapaces de adaptarse a nuevos patrones de comportamiento.
* **Regresión Logística:** Asume linealidad en los datos, limitando la captura de patrones complejos.
* **Red Neuronal (MLP):** Permite aproximar funciones no lineales complejas mediante capas ocultas y funciones de activación no lineales.

---

## 3. Objetivos e Indicadores (SMART)

* **Objetivo General:** Diseñar, entrenar y evaluar un Perceptrón Multicapa (MLP) para clasificar el riesgo de mora a 90 días mediante un pipeline reproducible en GitHub.
* **Objetivos Específicos:**
  1. Preprocesar y escalar las variables financieras de entrada para asegurar la estabilidad del gradiente.
  2. Construir la arquitectura de la red definiendo la cantidad de capas, funciones de activación, función de pérdida y algoritmo de optimización.
  3. Validar la convergencia del entrenamiento reduciendo la función de pérdida (Binary Cross-Entropy).

| Fase | Indicador | Meta | Evidencia |
| :--- | :--- | :--- | :--- |
| **Problema** | Documentación de contexto y justificación | 100% documentado | `README.md` |
| **Datos** | Ausencia de nulos y escalamiento | 0% faltantes, datos escalados | `notebooks/01_datos.ipynb` |
| **Arquitectura** | Compatibilidad de tensores | Verificación exitosa | `notebooks/04_architecture.ipynb` |
| **Entrenamiento** | Convergencia del error | Pérdida en descenso continuo | `notebooks/06_training.ipynb` |

---

## 4. Datos y Variables de Entrada

El dataset utiliza datos tabulares sobre el perfil financiero del solicitante:

| Variable | Tipo | Descripción | Target |
| :--- | :--- | :--- | :--- |
| `Ingreso_mensual` | Numérica | Ingresos netos mensuales del cliente (Normalizado) | No |
| `Indice_endeudamiento` | Numérica | Proporción de ingresos comprometidos en deudas | No |
| `Numero_retrasos_previos` | Categórica / Entero | Historial de moras en los últimos 24 meses | No |
| `Antiguedad_cliente_meses` | Numérica | Tiempo de permanencia del cliente en la entidad | No |
| `Mora_90_dias` | Binaria (0 / 1) | **Target:** 1 = Entra en mora, 0 = Paga a tiempo | **Sí** |

---

## 5. Fundamentos Matemáticos y Arquitectura de la Red

### Formulación Matemática
Cada neurona en la red calcula la combinación lineal de las entradas seguida de una transformación no lineal:
* **Combinación lineal:** $z = \sum_{i=1}^{n} (w_i \cdot x_i) + b = w^T x + b$
* **Activación:** $a = f(z)$

### Arquitectura Seleccionada (Perceptrón Multicapa)
* **Capa de Entrada:** 4 neuronas (correspondientes a las 4 variables de entrada $X$).
* **Capa Oculta 1:** 8 neuronas con activación **ReLU**.
* **Capa Oculta 2:** 4 neuronas con activación **ReLU**.
* **Capa de Salida:** 1 neurona con activación **Sigmoid** (salida probabilística entre 0 y 1).

---

## 6. Configuración del Aprendizaje y Optimización

* **Función de Activación Oculta (ReLU):** $f(z) = \max(0, z)$. Evita la saturación del gradiente en capas profundas y acelera la convergencia.
* **Función de Activación de Salida (Sigmoid):** $f(z) = \frac{1}{1 + e^{-z}}$. Mapea la salida en la escala continua $[0, 1]$ para interpretar la predicción como probabilidad.
* **Función de Pérdida (Loss):** **Binary Cross-Entropy (BCE)**.
  $$\mathcal{L}(y, \hat{y}) = - \Big[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \Big]$$
  Penaliza de forma estricta las predicciones confiables pero erróneas.
* **Algoritmo de Optimización:** **Adam** (Adaptive Moment Estimation).
  Ajusta dinámicamente la tasa de aprendizaje por cada peso basándose en los momentos de primer y segundo orden del gradiente.
* **Hiperparámetros Iniciales:**
  * Learning rate ($\eta$): `0.001`
  * Batch size: `32`
  * Epochs: `50`

---

## 7. Estructura de la Secuencia de Notebooks

1. `01_datos.ipynb`: Carga y exploración inicial del dataset.
2. `02_quality.ipynb`: Limpieza, tratamiento de faltantes y escalamiento de datos.
3. `03_fundamentos.ipynb`: Demostración matemática del cálculo forward de la neurona.
4. `04_architecture.ipynb`: Definición del modelo en TensorFlow/PyTorch y verificación de dimensiones.
5. `05_forward.ipynb`: Simulación del paso hacia adelante con tensores de prueba.
6. `06_training.ipynb`: Ciclo de entrenamiento, compilación y curvas de aprendizaje.
