# FER-models - Facial Emotion Recognition Experiments

Este repositorio contiene el código utilizado para el entrenamiento y evaluación de modelos de **Reconocimiento de Expresiones Faciales (FER)**, desarrollado como parte de un estudio experimental presentado en un **conference paper**.

El trabajo se centra en analizar el impacto de distintas **estrategias de entrenamiento y Knowledge Distillation (KD)** sobre modelos livianos, evaluando rendimiento, estabilidad y calibración en escenarios multi-dataset.

---

## Objetivo del proyecto

Explorar y comparar configuraciones de entrenamiento para FER mediante:

- **Fine-tuning directo** de modelos preentrenados.
- **Knowledge Distillation** con diferentes formulaciones.
- Evaluación consistente sobre múltiples datasets estándar de FER.

El foco del repositorio es **el pipeline experimental y la reproducibilidad del método**, no la distribución de resultados finales ni modelos entrenados.

---

## Alcance

Este repositorio incluye:
- Código de entrenamiento y evaluación.
- Configuraciones experimentales versionadas.
- Scripts auxiliares para inspección e inferencia.

Este repositorio **no incluye**:
- Datasets.
- Modelos entrenados (pesos).
- Resultados cuantitativos finales ni figuras.

---

## Arquitectura del código

El proyecto está organizado de forma modular, orientado a experimentación reproducible:

<img width="283" height="595" alt="image" src="https://github.com/user-attachments/assets/9e13936a-b2b7-4038-a1fb-09b7076c0e7e" />



Cada experimento está definido por:
- Scripts de entrenamiento específicos.
- Configuraciones independientes.
- Métricas y artefactos generados externamente.

---

## Experimentos implementados

Se implementaron **cinco configuraciones experimentales**:

- **Experimento 1**  
  Fine-tuning de modelos livianos sin Knowledge Distillation.

- **Experimento 2**  
  Fine-tuning con Knowledge Distillation básica aplicada a MobileNetV3 y EfficientNet.Se usaron 2 teachers en 2 fases: resnet50 e inceptionv3. Esto mostró saturación durante el aprendizaje, se corrigió para los experimentos 4 y 5.

- **Experimento 3**  
  Entrenamiento directo sin KD como baseline controlado.

- **Experimento 4**  
  Knowledge Distillation relacional basada en distancias, con modelo docente único.

- **Experimento 5**  
  Knowledge Distillation híbrida, combinando destilación por respuesta y destilación relacional.

El diseño experimental, hipótesis y resultados se describen en detalle en el artículo asociado.

---

## Datasets

Los experimentos utilizan datasets estándar de FER, entre ellos:

- AffectNet  
- FER-2013  
- RAF-DB  
- CK+

Por razones de licencia y tamaño, **los datasets no se incluyen en este repositorio**.

Los experimentos fueron ejecutados principalmente en **Kaggle**, con los datasets previamente cargados en el entorno.  
Las rutas a los datasets se definen en los archivos de configuración dentro de `configs/`.

---

## Instalación

### Requisitos
- Python 3.9 o superior
- pip

### Instalación de dependencias
pip install -r requirements.txt


---

El entrenamiento y la evaluación se ejecutan desde los scripts definidos en `src/`, utilizando las configuraciones correspondientes.

---

## Reproducibilidad

- Seeds fijados para Python, NumPy y PyTorch.
- Configuraciones experimentales versionadas.
- Código desacoplado por experimento.
- Pipeline diseñado para ejecución multi-dataset.

---

## Disponibilidad de resultados

Los resultados cuantitativos, figuras comparativas y modelos entrenados se proporcionan en el **conference paper asociado** y no forman parte de este repositorio.

---

## Notas finales

Este repositorio está orientado a **investigación académica y experimentación**, no a despliegue en producción.

Para el análisis completo de resultados y conclusiones, consultar la publicación asociada a este trabajo.


