# Segmentación de Nervios Periféricos en Ultrasonido (Dataset NerveUTP) - Espacio de Kreïn

Este repositorio contiene la rama principal del código fuente para el trabajo de grado *"Segmentación de Estructuras Nerviosas en Ultrasonido mediante Redes Neuronales en el Espacio de Kreïn"*. 

A diferencia del experimento del plexo braquial, este submódulo se enfoca en la base de datos **NerveUTP**, evaluando la capacidad de la arquitectura propuesta para adaptarse a estructuras de distinto calibre (nervios ciático, femoral, mediano y cubital) y lidiar con la severa degradación acústica (*speckle*) inherente a la anestesia regional.

## 🧠 El Desafío Topológico y la Solución

Las redes neuronales euclidianas tradicionales tienden a sobre-segmentar los nervios debido a la falta de límites claros frente a las fascias musculares (falsos positivos). 

Este proyecto propone una **U-Net en el Espacio de Kreïn** ($\mathcal{K} = \mathcal{H}^+ \oplus \mathcal{H}^-$), la cual evalúa simultáneamente:
1. **Energía Positiva (RBF):** Atrae la predicción hacia el patrón fascicular en "panal de abejas".
2. **Energía Negativa (Polinomial):** Ejerce una repulsión matemática contra el ruido de fondo.

**Hallazgo Principal:** El modelo de Kreïn alcanzó la Especificidad más alta del estudio (**0.9906 global**, llegando a **0.9928** en el difícil nervio cubital), demostrando ser una arquitectura inherentemente más segura para el entorno clínico al preferir el rechazo de ruido sobre la hiper-segmentación.

## 🛠️ Estructura del Repositorio y Scripts

### 1. Modelos Base
* `model.py`: U-Net Euclidiana estándar (Baseline).
* `hyperbolic_model.py`: U-Net proyectada en la Bola de Poincaré ($c=0.1$).
* `krein_model.py`: U-Net con clasificador en geometría indefinida de Kreïn.

### 2. Entrenamiento y Evaluación
* `train_krein.py`, `train.py`, `train_hyperbolic.py`: Scripts de entrenamiento utilizando la función de pérdida híbrida (`KreinDiceBCELoss` para el modelo principal).
* `test_final.py`: Orquestador de pruebas que evalúa las tres arquitecturas sobre el conjunto de prueba inédito (15% del dataset).
* `evaluate_nerves.py`: Script de evaluación estratificada para desglosar el rendimiento (Dice, IoU, Sensibilidad, Especificidad) según el tipo de nervio específico (Ciático, Femoral, Mediano, Cubital).

### 3. Interpretabilidad y Visualización
* `visualize_krein.py` (y variantes): Genera los paneles visuales comparando la Verdad Terreno (verde) contra la Predicción (rojo) y extrae los mapas de energía topológica subyacentes.
* `m_dropout_analysis.py`: **Script de Interpretabilidad Bayesiana.** Inyecta abandono espacial (*Monte Carlo Dropout*, $p=0.3$, $N=150$) en tiempo de inferencia para aislar la incertidumbre epistémica. Calcula la Correlación de Pearson ($r$) para demostrar matemáticamente que la alta energía negativa del modelo suprime activamente la duda algorítmica en el ruido de fondo.

## 🚀 Uso y Replicación

**1. Preparar el entorno:**
Asegúrate de configurar la constante `DATA_DIR` dentro de los scripts apuntando a tu ruta local del dataset NerveUTP.

**2. Ejecutar la evaluación global:**
```bash
python test_final.py
