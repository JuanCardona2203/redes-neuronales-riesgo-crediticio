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
| **Softmax** | $\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum e^{z_j}}$ | $[0, 1]$ | Genera una distribución de probabilidad mult
