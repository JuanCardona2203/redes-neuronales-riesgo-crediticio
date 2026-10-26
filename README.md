# Predicción de Riesgo Crediticio mediante Redes Neuronales (MLP)

| Dato | Registro del Estudiante / Proyecto |
| :--- | :--- |
| **Estudiante / equipo** | Yenifer Salazar Restrepo - Juan Esteban Cardona Jaramillo - Camilo Andrés Estrada Sosa |
| **Programa / área** | Ingeniería / Ciencia de Datos / Redes Neuronales |
| **Proyecto** | Clasificación de Riesgo Crediticio y Predicción de Mora con Perceptrón Multicapa (MLP) |
| **Docente** | Docente Titular de Redes Neuronales |
| **Línea / semillero / investigación** | Inteligencia Computacional y Analítica Financiera |
| **Versión / fecha** | v1.0 — Octubre 2026 |
| **Repositorio / evidencia** | [GitHub - redes-neuronales-riesgo-crediticio](https://github.com/JuanCardona2203/redes-neuronales-riesgo-crediticio) |
| **Estado** | Implementación / Validación (Fases 1 a 10) |

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
