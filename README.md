# Predicción de Riesgo Crediticio mediante Redes Neuronales (MLP)

| Dato | Registro del Estudiante / Proyecto |
| :--- | :--- |
| **Estudiante / equipo** | Yenifer Salazar Restrepo - Juan Esteban Cardona Jaramillo - Camilo Andrés Estrada Sosa |
| **Programa / área** | Ingeniería / Ciencia de Datos / Redes Neuronales |
| **Proyecto** | Clasificación de Riesgo Crediticio y Predicción de Mora con Perceptrón Multicapa (MLP) |
| **Docente** | Feibert Alirio Guzmán Pérez |
| **Línea / semillero / investigación** | Inteligencia Computacional y Analítica Financiera |
| **Versión / fecha** | v1.0 — Octubre 2026 |
| **Repositorio / evidencia** | [GitHub - redes-neuronales-riesgo-crediticio](https://github.com/JuanCardona2203/redes-neuronales-riesgo-crediticio) |
| **Estado** | Final (Fases 1 a 28) |

---

## Propósito del Proyecto
Transformar la evaluación de riesgo crediticio en un modelo reproducible de aprendizaje profundo, partiendo desde los fundamentos matemáticos de la neurona artificial hasta el diseño, propagación hacia adelante, optimización y entrenamiento de un Perceptrón Multicapa (MLP).

---

## 1. Definición del Problema

| Elemento | Registro del Proyecto |
| :--- | :--- |
| **Contexto** | Evaluación de solicitudes de créditos de consumo en entidades financieras. |
| **Problema observable** | Ineficiencia y tasa elevada de cartera vencida (>90 días) por modelos rígidos de aprobación. |
| **Antecedentes** | Los scorecards tradicionales basados en sumatorias de puntos o regresión lineal no capturan interacciones complejas ni fronteras de decisión no lineales. |
| **Causas** | Cambios en el perfil socioeconómico de los solicitantes e interacciones complejas entre el ingreso, la tasa y el sobreendeudamiento. |
| **A quién impacta** | Al departamento de riesgos del banco (pérdidas de capital) y a los clientes solventes (rechazos injustificados). |
| **Consecuencia** | Aumento de las provisiones por incobrabilidad y pérdida de competitividad en el mercado crediticio. |
| **Decisión a mejorar** | Clasificación automática y probabilística sobre aprobar o denegar una solicitud de crédito. |
| **Oportunidad con redes neuronales** | Capacidad del Perceptrón Multicapa (MLP) para aprender funciones no lineales continuas a partir de múltiples variables financieras. |
| **Restricciones** | Mantenimiento estricto de la reproducibilidad, control de sesgos, ausencia de data leakage y normalización de entradas. |

### 1.1 Pregunta de Investigación
¿En qué medida la implementación de un Perceptrón Multicapa (MLP) optimizado mediante el algoritmo Adam permite clasificar con mayor precisión la mora crediticia a 90 días en comparación con los enfoques estadísticos lineales convencionales?

### 1.2 Justificación Técnica

| Alternativa | Ventaja | Limitación | Razón para utilizar Red Neuronal |
| :--- | :--- | :--- | :--- |
| **Reglas de negocio** | Altamente interpretables y fáciles de implementar en sistemas legacy. | Estáticas, rígidas, incapacidad de adaptarse a nuevos patrones de fraude o mora. | La RN ajusta sus pesos automáticamente mediante datos históricos sin reglas manuales. |
| **Regresión / estadística** | Modelos simples, bajo costo computacional, salida probabilística directa. | Asumen linealidad estricta e independencia entre variables explicativas. | La RN aprende combinaciones no lineales complejas entre el nivel de endeudamiento y la tasa. |
| **ML convencional (Dec. Trees)** | Maneja bien datos tabulares y relaciones no lineales simples. | Propenso al sobreajuste si no se podan correctamente los árboles. | Sirve como baseline de referencia; el MLP permite construir representaciones internas más ricas. |
| **Red Neuronal (MLP)** | Alta capacidad de aproximación universal de funciones no lineales. | Requiere ajuste riguroso de hiperparámetros y normalización cuidadosa. | **Seleccionada:** Ofrece la flexibilidad requerida para modelar fronteras de decisión complejas en riesgo financiero. |

---

## 2. Objetivos e Indicadores

### Objetivo General
Diseñar, fundamentar e implementar una red neuronal densa (MLP) para predecir la probabilidad de incumplimiento de pago (mora a 90 días) utilizando el dataset *Credit Risk*, garantizando reproducibilidad técnica en GitHub.

### Objetivos Específicos
1. **Preprocesamiento:** Limpiar, imputar y escalar min-max/estándar las variables financieras de entrada para garantizar la estabilidad de los gradientes.
2. **Modelado y Arquitectura:** Estructurar un MLP definiendo capas densas, funciones de activación (ReLU/Sigmoid), función de pérdida (*Binary Cross-Entropy*) y optimizador (*Adam*).
3. **Optimización y Validación:** Ejecutar experimentos de entrenamiento verificando matemáticamente el *Forward Pass*, la reducción del error y la actualización de pesos por *Backpropagation*.

### Matriz de Seguimiento (SMART)

| Fase | Evidencia | Indicador | Meta | Fecha / Horizonte |
| :--- | :--- | :--- | :--- | :--- |
| **Problema** | `README.md` | Formulación completa y matriz comparativa | 100% documentado | Semana 1 |
| **Datos** | `notebooks/01_datos.ipynb` / `02_quality.ipynb` | Tratamiento de nulos, outliers y escalamiento | 0% faltantes, $X \in [0,1]$ | Semana 2 |
| **Arquitectura** | `notebooks/04_architecture.ipynb` | Diagrama y compatibilidad de tensores | Verificación exitosa | Semana 3 |
| **Modelo** | `notebooks/06_training.ipynb` | Descenso de la pérdida (*Loss*) en entrenamiento | Convergencia de curva | Semana 4 |

---

## 3. Datos y Variables de Entrada

### Fuentes de Información
| Fuente | Formato | Frecuencia | Licencia / Acceso | Responsable |
| :--- | :--- | :--- | :--- | :--- |
| **Kaggle Credit Risk Dataset** | CSV | Estática | CC0: Public Domain | Equipo de Desarrollo |

### 3.1 Diccionario de Datos

| Variable | Tipo | Descripción | Rango / Clases | Target | Origen | Transformación |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `person_age` | Entero | Edad del solicitante en años | 20 – 144 | No | Directo | Filtro de outliers (>100) y StandardScale |
| `person_income` | Continuo | Ingreso anual declarado (USD) | 4,000 – 6,000,000 | No | Directo | Log-transform + MinMaxScaler |
| `loan_amnt` | Continuo | Monto del préstamo solicitado (USD) | 500 – 35,000 | No | Directo | MinMaxScaler $[0, 1]$ |
| `loan_int_rate` | Continuo | Tasa de interés asignada (%) | 5.42% – 23.22% | No | Directo | Imputación por mediana + MinMaxScaler |
| `loan_percent_income` | Continuo | Proporción cuota/ingreso anual | 0.00 – 0.83 | No | Derivado | Validado directo en escala $[0, 1]$ |
| `cb_person_cred_hist_length` | Entero | Años de historial crediticio | 2 – 30 | No | Directo | MinMaxScaler $[0, 1]$ |
| `loan_status` | Binario | Estado de pago del crédito | 0 (Cumple), 1 (Default/Mora) | **Sí** | Target | Mantenido como entero binario $\{0, 1\}$ |

### 3.2 Calidad y Gobernanza de Datos
- [x] **Faltantes:** Imputación de la tasa de interés mediante la mediana estratificada por grado de riesgo.
- [x] **Outliers:** Eliminación de registros con edad inconsistente (>100 años).
- [x] **Data Leakage:** División estricta de conjuntos (Train/Test) *antes* de calcular los parámetros de escalamiento (Min/Max).
- [x] **Desbalance:** Evaluación de la proporción de clases ($\approx 21.8\%$ positivos / default).

---

## 4. Fundamentos Matemáticos de la Neurona

Formulación para una neurona individual en la red:

| Concepto | Definición / Fórmula | Aplicación en el Proyecto |
| :--- | :--- | :--- |
| **Entrada $x$** | $\mathbf{x} = [x_1, x_2, \dots, x_6]^T$ | Vector con las 6 variables financieras escaladas del solicitante. |
| **Pesos $w$** | $\mathbf{w} = [w_1, w_2, \dots, w_6]^T$ | Importancia o ponderación asignada a cada variable financiera. |
| **Sesgo $b$** | $b \in \mathbb{R}$ | Umbral de activación independiente de las entradas. |
| **Combinación lineal** | $z = \mathbf{w}^T \mathbf{x} + b = \sum_{i=1}^{n} (w_i x_i) + b$ | Suma ponderada de perfil del cliente. |
| **Activación** | $a = f(z)$ | Transformación no lineal de la combinación lineal. |
| **Salida $\hat{y}$** | $\hat{y} = a^{(L)}$ | Probabilidad predicha de que el cliente caiga en mora ($[0, 1]$). |
| **Error / Loss** | $\mathcal{L}(y, \hat{y})$ | Métrica de discrepancia entre el estado real $y$ y la predicción $\hat{y}$. |

### 4.1 Ejemplo Numérico Verificable
Supongamos un cliente con entradas normalizadas $\mathbf{x} = [0.25, 0.40, 0.30, 0.15]$, pesos $\mathbf{w} = [0.80, 1.20, -0.50, 0.60]$ y sesgo $b = -0.10$:

| Entrada ($x_i$) | Peso ($w_i$) | Producto ($x_i \cdot w_i$) | Sesgo ($b$) | Suma $z$ | Activación $f(z)$ (Sigmoid) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $x_1 = 0.25$ | $w_1 = 0.80$ | $0.200$ | — | — | — |
| $x_2 = 0.40$ | $w_2 = 1.20$ | $0.480$ | — | — | — |
| $x_3 = 0.30$ | $w_3 = -0.50$ | $-0.150$ | — | — | — |
| $x_4 = 0.15$ | $w_4 = 0.60$ | $0.090$ | — | — | — |
| **Total** | — | **$\sum = 0.620$** | **$-0.100$** | **$z = 0.520$** | $\sigma(0.520) = \frac{1}{1 + e^{-0.520}} \approx \mathbf{0.6271}$ |

---

## 5. Funciones de Activación

| Función | Fórmula | Rango | Ventaja | Limitación | Uso en el Proyecto |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sigmoid** | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | $(0, 1)$ | Mapea la salida directamente a una probabilidad. | Sufre de gradiente desvaneciente en valores extremos. | **Capa de Salida** |
| **Tanh** | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1, 1)$ | Centrada en cero, facilita la convergencia frente a sigmoid. | Mantiene problema de gradiente desvaneciente. | No utilizada |
| **ReLU** | $f(z) = \max(0, z)$ | $[0, \infty)$ | Computacionalmente eficiente, no se satura para $z > 0$. | "Dying ReLU" si $z < 0$ constantemente. | **Capas Ocultas** |
| **Leaky ReLU** | $f(z) = \max(\alpha z, z)$ | $(-\infty, \infty)$ | Evita el problema de neuronas muertas activando con $\alpha = 0.01$. | Requiere ajustar el hiperparámetro $\alpha$. | Alternativa experimental |
| **Softmax** | $\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum e^{z_j}}$ | $[0, 1]$ | Genera una distribución de probabilidad multiclase. | Ineficiente para problemas estrictamente binarios. | No utilizada |

### 5.1 Selección Justificada de Activaciones
* **Capas Ocultas (ReLU):** Seleccionada por acelerar la velocidad de convergencia del gradiente y permitir que la red aprenda fronteras complejas sin costo computacional elevado.
* **Capa de Salida (Sigmoid):** Indispensable para problemas de clasificación binaria, transformando la salida final en una probabilidad acotada en $[0, 1]$.

---

## 6. Arquitectura de la Red (MLP)

| Capa | Tipo | Neuronas | Entrada | Salida | Activación | Propósito |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Entrada** | Input | 6 | $(N, 6)$ | $(N, 6)$ | Ninguna | Recibir las 6 variables financieras escaladas. |
| **Oculta 1** | Dense | 16 | $(N, 6)$ | $(N, 16)$ | ReLU | Extraer patrones de interacción primaria. |
| **Oculta 2** | Dense | 8 | $(N, 16)$ | $(N, 8)$ | ReLU | Condensar representaciones no lineales secundarias. |
| **Salida** | Dense | 1 | $(N, 8)$ | $(N, 1)$ | Sigmoid | Generar probabilidad final de mora crediticia. |

### 6.1 Familias de Redes Analizadas
* **Perceptrón de una capa:** Limitado a separabilidad lineal (insuficiente para riesgo bancario).
* **MLP (Perceptrón Multicapa):** **Seleccionado.** Ideal para datos tabulares estructurados con variables cuantitativas.
* **CNN / RNN / LSTM:** No aplican debido a que no estamos procesando imágenes ni series temporales secuenciales explícitas.

### 6.2 Diagrama de Flujo Arquitectónico
```text
[Entradas: 6 variables] 
         │
         ▼
 ┌──────────────┐
 │ Capa Oculta 1│  (16 Neuronas + Activation ReLU)
 └───────┬──────┘
         │
         ▼
 ┌──────────────┐
 │ Capa Oculta 2│  (8 Neuronas + Activation ReLU)
 └───────┬──────┘
         │
         ▼
 ┌──────────────┐
 │ Capa Salida  │  (1 Neurona + Activation Sigmoid) ──► Output: P(Mora) ∈ [0, 1]
 └──────────────┘
```

## 7. Forward Propagation (Paso Hacia Adelante)

| Paso | Operación Matemática | Entrada | Parámetros / Dimensiones | Activación | Salida / Dimensión |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | $z^{(1)} = X W^{(1)} + b^{(1)}$ | $X \in \mathbb{R}^{B \times 6}$ | $W^{(1)} \in \mathbb{R}^{6 \times 16}, b^{(1)} \in \mathbb{R}^{16}$ | ReLU | $A^{(1)} \in \mathbb{R}^{B \times 16}$ |
| **2** | $z^{(2)} = A^{(1)} W^{(2)} + b^{(2)}$ | $A^{(1)} \in \mathbb{R}^{B \times 16}$ | $W^{(2)} \in \mathbb{R}^{16 \times 8}, b^{(2)} \in \mathbb{R}^{8}$ | ReLU | $A^{(2)} \in \mathbb{R}^{B \times 8}$ |
| **3** | $z^{(3)} = A^{(2)} W^{(3)} + b^{(3)}$ | $A^{(2)} \in \mathbb{R}^{B \times 8}$ | $W^{(3)} \in \mathbb{R}^{8 \times 1}, b^{(3)} \in \mathbb{R}^{1}$ | Sigmoid | $\hat{Y} \in \mathbb{R}^{B \times 1}$ |

### 7.1 Verificación Dimensional
Se verifica que el producto de matrices mantenga consistencia dimensional para un tamaño de lote (batch size) de $B = 32$:
* **Capa Oculta 1:** $(32 \times 6) \cdot (6 \times 16) + (16) \implies (32 \times 16)$
* **Capa Oculta 2:** $(32 \times 16) \cdot (16 \times 8) + (8) \implies (32 \times 8)$
* **Capa de Salida:** $(32 \times 8) \cdot (8 \times 1) + (1) \implies (32 \times 1)$

---

## 8. Función de Pérdida (Loss)

| Tarea | Función Candidata | Fórmula | Supuesto | ¿Se Utiliza? |
| :--- | :--- | :--- | :--- | :---: |
| **Clasificación Binaria** | **Binary Cross-Entropy (BCE)** | $\mathcal{L} = - [y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$ | Target $y \in \{0, 1\}$, salida $\hat{y} \in (0, 1)$. | **Sí** |
| **Regresión** | Mean Squared Error (MSE) | $\mathcal{L} = \frac{1}{2} (y - \hat{y})^2$ | Salidas continuas no acotadas. | No |
| **Multiclase** | Categorical Cross-Entropy | $\mathcal{L} = -\sum y_i \log(\hat{y}_i)$ | Mutuamente excluyente con $K > 2$ clases. | No |

### 8.1 Interpretación del Error
La función Binary Cross-Entropy penaliza fuertemente a la red cuando realiza predicciones seguras pero incorrectas (por ejemplo, asignar $\hat{y} = 0.99$ cuando la etiqueta real es $y = 0$). Una disminución continua en el valor del error (*loss*) indica que las probabilidades asignadas a la clase correcta aumentan progresivamente durante el entrenamiento.

---

## 9. Backpropagation y Aprendizaje

| Concepto | Descripción / Ecuación | Registro en el Proyecto |
| :--- | :--- | :--- |
| **Loss General** | $J(W, b) = \frac{1}{B} \sum_{i=1}^{B} \mathcal{L}(y^{(i)}, \hat{y}^{(i)})$ | Promedio del error en el batch actual. |
| **Gradiente** | $\nabla_{W} J = \frac{\partial J}{\partial W}$ | Vector de derivadas parciales que indica la dirección de máximo crecimiento del error. |
| **Regla de actualización** | $W \leftarrow W - \eta \frac{\partial J}{\partial W}$ | Ajuste de pesos en la dirección opuesta al gradiente. |
| **Learning Rate ($\eta$)** | $\eta = 0.001$ | Magnitud del paso en cada actualización de parámetros. |
| **Epoch** | 1 pasada completa sobre el conjunto de entrenamiento | Configurado inicialmente a 50 épocas. |
| **Batch Size** | Subconjunto de datos procesado simultáneamente | Configurado a 32 muestras. |

### 9.1 Cadena de Derivadas (Regla de la Cadena)
Para actualizar los pesos de la capa de salida $W^{(3)}$:
$$\frac{\partial J}{\partial W^{(3)}} = \frac{\partial J}{\partial \hat{Y}} \cdot \frac{\partial \hat{Y}}{\partial z^{(3)}} \cdot \frac{\partial z^{(3)}}{\partial W^{(3)}}$$

Para la combinación de Binary Cross-Entropy y activación Sigmoid, la derivada simplificada del error respecto a la entrada lineal de la última capa es:
$$\frac{\partial J}{\partial z^{(3)}} = \hat{Y} - Y$$

---

## 10. Algoritmo de Optimización e Hiperparámetros

| Optimizador | Idea Clave | Ventaja | Limitación | ¿Se Prueba? |
| :--- | :--- | :--- | :--- | :---: |
| **Gradient Descent** | Actualización usando todo el dataset a la vez. | Estable y exacto. | Lento e inviable en datasets grandes. | No |
| **SGD** | Actualización muestra por muestra o minibatch. | Rápido, escapa de mínimos locales. | Alta oscilación durante el descenso. | No |
| **Adam** | Combina Momentum con RMSProp (Momentos adaptativos $m_t$ y $v_t$). | Robusto, rápido, ajusta la tasa por parámetro. | Puede requerir weight decay para prevenir sobreajuste. | **Sí (Principal)** |

### 10.1 Configuración de Hiperparámetros

| Parámetro | Valor Inicial | Rango / Alternativas | Justificación Técnica |
| :--- | :---: | :---: | :--- |
| **Learning Rate ($\eta$)** | `0.001` | $[0.0001, 0.01]$ | Tasa estándar para Adam que asegura estabilidad inicial. |
| **Batch Size** | `32` | $[16, 64]$ | Balance óptimo entre velocidad de ejecución y estabilidad de gradientes. |
| **Epochs** | `50` | $[30, 100]$ | Cantidad suficiente para observar la convergencia del error. |
| **Betas ($\beta_1, \beta_2$)** | `(0.9, 0.999)` | Valores estándar de Adam | Estimación de primer y segundo momento de los gradientes. |
| **Seed (Semilla)** | `42` | Fijo | Garantiza la reproducibilidad exacta de los experimentos. |

---

## 11. Entrenamiento y Validación

Evidencia: `notebooks/06_training.ipynb`

El modelo se entrena sobre los datos escalados generados en `02_quality.ipynb`. El conjunto de entrenamiento se divide internamente en Train/Validación (80/20, estratificado), reservando el conjunto de Test para la evaluación final.

| Parámetro | Valor | Justificación |
| :--- | :--- | :--- |
| **Train/Validación** | 80% / 20% | División estratificada manteniendo la proporción de clases. |
| **Test** | Reservado | No se usa en entrenamiento; se evalúa en la Sección 14. |
| **Seed** | `42` | Reproducibilidad exacta. |
| **Función de pérdida** | Binary Cross-Entropy | Clasificación binaria con salida Sigmoid. |
| **Optimizador** | Adam (lr=0.001) | Momentos adaptativos. |
| **Guardado** | `models/mlp_credit_risk.pt` | Pesos del modelo entrenado. |

**Resultado:** la pérdida desciende de forma estable (train ≈ 0.39 → 0.36; validación ≈ 0.39 → 0.36), confirmando que la red aprende. Las curvas de train y validación descienden juntas, sin señales de sobreajuste.

---

## 12. Regularización y Generalización

Evidencia: `notebooks/07_regularization.ipynb`

| Técnica | Problema que aborda | Aplicación | Resultado |
| :--- | :--- | :--- | :--- |
| **Dropout** | Sobreajuste | Capas ocultas (0.3 / 0.5) | No mejora: el modelo base no sobreajusta. |
| **Weight decay (L2)** | Sobreajuste | Adam (`weight_decay` 1e-4 / 1e-3) | Incrementa ligeramente la pérdida. |
| **Early stopping** | Sobreajuste | Monitoreo de val_loss (patience=5) | Sin efecto relevante. |

**Conclusión:** como las curvas de entrenamiento y validación ya convergen juntas, no existe sobreajuste que corregir. Las técnicas de regularización se evalúan y se documentan como alternativas descartadas con evidencia, conservando la arquitectura base.

---

## 13. Cuaderno de Experimentos

Evidencia: `notebooks/07_regularization.ipynb` · `notebooks/reports/experimentos.csv` · `docs/EXPERIMENTOS.md`

Experimentos controlados (semilla y validación fijas). Métrica: pérdida de validación (BCE) final.

| ID | Cambio | Val Loss (BCE) | Decisión |
| :--- | :--- | :---: | :--- |
| **EXP-01** | Baseline MLP (6-16-8-1) | **0.3583** | **Seleccionado** |
| EXP-07 | Batch size 16 | 0.3604 | Descartado |
| EXP-08 | Early stopping (patience=5) | 0.3619 | Descartado |
| EXP-02 | Dropout 0.3 | 0.3624 | Descartado |
| EXP-06 | Dropout 0.3 + WD 1e-4 | 0.3626 | Descartado |
| EXP-04 | Weight decay 1e-4 | 0.3684 | Descartado |
| EXP-05 | Weight decay 1e-3 | 0.3697 | Descartado |
| EXP-03 | Dropout 0.5 | 0.3841 | Descartado |

---

## 14. Evaluación y Análisis de Errores

Evidencia: `notebooks/08_evaluation.ipynb` · `notebooks/reports/comparacion_modelos.csv`

Métricas del MLP sobre el conjunto de Test reservado (6.516 muestras):

| Métrica | Valor | Interpretación |
| :--- | :---: | :--- |
| **Accuracy** | 0.843 | Aciertos globales. |
| **Precision** | 0.671 | De los clientes marcados como mora, cuántos lo son realmente. |
| **Recall** | 0.548 | De los morosos reales, cuántos detecta el modelo. |
| **F1** | 0.603 | Equilibrio entre precision y recall. |
| **ROC-AUC** | 0.851 | Capacidad de separar las clases. |

### 14.1 Matriz de Confusión

| | Predicho Cumple | Predicho Mora |
| :--- | :---: | :---: |
| **Real Cumple** | 4712 (TN) | 382 (FP) |
| **Real Mora** | 643 (FN) | 779 (TP) |

| Error | Cantidad | Impacto en el negocio |
| :--- | :---: | :--- |
| **Falsos Positivos** | 382 | Clientes solventes rechazados (pérdida de negocio). |
| **Falsos Negativos** | 643 | Morosos aprobados (pérdida de capital). Principal punto a mejorar. |

El Recall moderado refleja el desbalance de clases (~21.8% de mora): el modelo aún deja pasar casi la mitad de los morosos, lo que se documenta como limitación.

---

## 16. Comparación de Redes y Selección Técnica

Evidencia: `notebooks/08_evaluation.ipynb`

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Regresión Logística (baseline) | 0.831 | 0.698 | 0.398 | 0.507 | 0.827 |
| **MLP (Red Neuronal)** | **0.843** | 0.671 | **0.548** | **0.603** | **0.851** |

**Selección:** el MLP supera al baseline lineal en Recall (+15 pts), F1 (+10 pts) y ROC-AUC (+2.4 pts). Al detectar más casos de mora real, responde afirmativamente a la pregunta de investigación: la red neuronal clasifica la mora con mayor precisión que el enfoque estadístico lineal convencional.

---

## 15. Interpretación de la Red (XAI)

Evidencia: `notebooks/09_interpretation.ipynb` · `notebooks/reports/importancia_variables.csv`

Se aplica **permutation importance** sobre el Test: se permuta cada variable y se mide la caída del ROC-AUC. Una caída mayor indica mayor importancia para la predicción.

| Variable | Importancia (caída ROC-AUC) | Lectura |
| :--- | :---: | :--- |
| `loan_percent_income` | 0.187 | Proporción cuota/ingreso: variable más influyente. |
| `loan_int_rate` | 0.140 | Tasa de interés asignada: segunda más influyente. |
| `loan_amnt` | 0.037 | Monto del préstamo. |
| `person_income` | 0.015 | Ingreso anual. |
| `person_age` | 0.0002 | Edad: influencia casi nula. |
| `cb_person_cred_hist_length` | 0.0001 | Historial crediticio: influencia casi nula. |

**Interpretación:** el modelo prioriza el nivel de endeudamiento relativo y la tasa de interés, patrones financieramente coherentes con el riesgo de mora. La edad y la antigüedad del historial resultan poco relevantes.

**Limitación:** la permutation importance mide influencia predictiva, **no causalidad**. Indica qué variables usa la red, no una relación causa-efecto.

---

## 17. Despliegue de la Red Neuronal

Evidencia: `notebooks/10_deployment.ipynb` · `src/predict.py` · `app/app.py`

| Componente | Alternativa elegida | Justificación |
| :--- | :--- | :--- |
| **Serialización** | PyTorch (`state_dict`) | Formato nativo del framework usado. |
| **Inferencia** | `src/predict.py` | Funciones reutilizables de carga y predicción. |
| **Interfaz** | Streamlit (`app/app.py`) | Demo sencilla para evaluar solicitudes. |

**Flujo de inferencia:** entrada → validación (variables completas y sin nulos) → escalamiento idéntico al entrenamiento (MinMax ajustado solo en Train) → red neuronal → probabilidad de mora → decisión (APROBAR / DENEGAR).

### 17.1 Casos de prueba

| Prueba | Entrada | Resultado | Estado |
| :--- | :--- | :--- | :---: |
| Caso válido | Perfil de bajo riesgo | Prob. mora 2.2% → APROBAR | OK |
| Caso límite | Alto endeudamiento (0.67) y tasa 20% | Prob. mora 99.1% → DENEGAR | OK |
| Entrada inválida | Falta una variable | Error controlado (ValueError) | OK |
| Error de inferencia | Valor nulo en una variable | Error controlado (ValueError) | OK |

### 17.2 Ejecución de la demo

```bash
streamlit run app/app.py
```

---

## 18. Reproducibilidad y Estructura del Repositorio

```text
redes-neuronales-riesgo-crediticio/
├── notebooks/        # 01 a 10: datos, fundamentos, arquitectura, entrenamiento, evaluación, despliegue
│   └── reports/      # métricas generadas (experimentos, comparación, importancia)
├── src/              # model.py (arquitectura) y predict.py (inferencia)
├── models/           # modelo serializado (se genera al entrenar)
├── app/              # demo Streamlit
├── docs/             # informe, experimentos, ética, uso de IA, ficha de entrega
├── presentation/     # diapositivas y video de sustentación
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

### Instalación y ejecución

```bash
pip install -r requirements.txt
```

Ejecutar los notebooks `01` a `10` en orden. Los notebooks funcionan tanto en local como en Google Colab (clonan el repositorio si es necesario). La semilla fija (42) garantiza reproducibilidad.

Demo opcional:

```bash
streamlit run app/app.py
```

---

## 19. Uso de IA · 20. Ética · 22. Informe

| Documento | Contenido |
| :--- | :--- |
| `docs/USO_IA.md` | Registro de prompts y validación humana (§19). |
| `docs/ETICA.md` | Riesgos, sesgos y mitigaciones (§20). |
| `docs/INFORME.md` | Informe técnico completo (§22). |
| `docs/ENTREGA_FINAL.md` | Ficha de entrega final (§28). |

---

## Limitaciones

- **Recall moderado (54.8%):** el modelo deja pasar casi la mitad de los morosos reales, por el desbalance de clases. Es la principal área de mejora.
- **Prototipo académico:** no está validado para uso en producción real.
- **Sin causalidad:** la interpretación muestra correlación con la predicción, no causas.

---

## 27. Bibliografía

| Tipo | Fuente |
| :--- | :--- |
| Dataset | Credit Risk (Kaggle, licencia CC0: Public Domain). |
| Framework | PyTorch — biblioteca de aprendizaje profundo. |
| Algoritmo | Kingma & Ba (2014), *Adam: A Method for Stochastic Optimization*. |
| Métricas | scikit-learn — métricas de clasificación. |
