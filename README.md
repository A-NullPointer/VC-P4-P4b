<div align="center">
  <img src="https://www.eii.ulpgc.es/sites/default/files/eii-acron-mod.png"
       alt="Logo ULPGC"
       width="500"
       style="margin-bottom: 10px;">
</div>

<h1 align="center">Práctica 4</h1>

<div align="center" style="font-family: 'Segoe UI', sans-serif; line-height: 1.6; margin-top: 30px;">
  <h2 style="font-size: 28px; margin-bottom: 10px;">
    Asignatura: <span>Visión por Computador</span>
  </h2>
  <p style="font-size: 18px; margin: 4px 0;">
    Grado en Ingeniería Informática
  </p>
  <p style="font-size: 18px; margin-top: 10px;">
    Curso <strong>2025 / 2026</strong>
  </p>
</div>

<h2 align="center">Autores</h2>

- Asmae Ez Zaim Driouch
- Javier Castilla Moreno

<h2 align="center">Bibliotecas utilizadas</h2>

[![NumPy](https://img.shields.io/badge/NumPy-%23013243?style=for-the-badge&logo=numpy)](https://numpy.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-%23FD8C00?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%43FF6400?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%230F4B8A?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Rtree](https://img.shields.io/badge/Rtree-%23009E73?style=for-the-badge&logo=python&logoColor=white)](https://rtree.readthedocs.io/)

## Cómo usar
### Primer paso: clonar este repositorio
```bash
git clone "https://github.com/A-NullPointer/VC-P5-P5"
```
### Segundo paso: Activar tu environment e instalar dependencias
> [!NOTE]
> Todas las dependencias pueden verse en [este archivo](environment.yml). Si se desea, puede crearse un entorno de Conda con dicho archivo.

Si se opta por crear un nuevo `Conda environment` a partir del archivo expuesto, es necesario abrir el `Anaconda Prompt` y ejecutar lo siguiente:

```bash
conda env create -f environment.yml
```

Posteriormente, se activa el entorno:

```bash
conda activate VC_P4
```

### Tercer paso: ejecutar el cuaderno
Finalmente, abriendo nuestro IDE favorito y teniendo instalado todo lo necesario para poder ejecutar notebooks, se puede ejecutar el cuaderno de la práctica [Practica3.ipynb](Practica3.ipynb) seleccionando el environment anteriormente creado.

> [!IMPORTANT]
> Todos los bloques de código deben ejecutarse en orden, de lo contrario, podría ocasionar problemas durante la ejecución del cuaderno.

<h1 align="center">Tareas</h1>

<h2 align="center">Parte 4: Detección y conteo de personas y vehículos</h2>

Para esta parte de la práctica, se ha realizado la detección de las personas y vehículos en el vídeo proporcionado para tal haciendo uso del modelo YOLO nano. Además, se ha entrenado este modelo para que pueda detectar matrículas, siendo posteriormente leídas por algún procesador de texto como puede ser Pytesseract y SmolVLM en este caso.

Por otro lado, se ha usado un dataset de cosecha propia para este entrenamiento, dicho dataset puede verse en la siguiente [carpeta de OneDrive](https://alumnosulpgc-my.sharepoint.com/:f:/g/personal/javier_castilla101_alu_ulpgc_es/EklVrCWbPCVKpoXbImvYWIwBm1HPPRCzuYsJKcC7jKnT3Q?e=LGbZZP).

Para etiquetar las imgánes del dataset, se ha usado la herramienta en la nube [makesense.ai](makesense.ai). Finalmente, se han dispuesto las imágenes y las etiquetas siguiendo la siguiente estructura de carpetas:
- dataset
  - train
    - images
    - labels
  - val
    - images
    - labels

Una vez el dataset ha sido creado y etiquetado, se propone el siguiente código para realizar el entrenamiento usando YOLO:

```python

```

A continuación, tras haber entrenado un modelo capaz de detectar matrículas, se dispone a la generación de un vídeo donde se detectarán personas y vehículos con sus respectivas matrículas a partir del vídeo proporcionado para esta práctica. En dicha generación, se mostrará además en tiempo real el conteo de cada clase detectada.

El modo de proceder para detectar la matrícula ha sido detectar primero el vehículo, y a partir del recorte del fotograma donde se ha detectado, usar nuestro modelo entrenado para detectar la matrícula. De este modo, nos resulta algo más fácil la detección, a diferencia de si se tratase de detectar a partir de la totalidad del fotograma. Una vez detectada la matrícula, se le pasa a una función que la procesa para conseguir su texto con EasyOCR, PyTesseract o SmolVLM.

Tras todo esto, los resultados son anotados en un archivo csv para cada fotograma acorde al siguiente formato:

```
fotograma, tipo_objeto, confianza, identificador_tracking, x1, y1, x2, y2, matrícula_en_su_caso, confianza, mx1,my1,mx2,my2, texto_matricula
```

>  [!NOTE]
> Las columnas x1, y1, x2, y2 corresponden al bounding box de la detección. De igual manera, mx1, my1, mx2, my2 delimitan la detección de la matrícula en su caso.

A continuación, se muestran partes de los vídeos generados con las lecturas de matrículas usando EasyOCR y SmolVLM respectivamente:

<h3 align="center">Fragmento usando EasyOCR</h3> 

<div style="text-align: center;" align="center"> <img src="salida_detecciones_easyocr.gif"> </div>

<h3 align="center">Fragmento usando SmolVLM</h3> 

<div style="text-align: center;" align="center"> <img src="salida_detecciones_smolvlm.gif"> </div>

Como se puede osbervar en los fragmentos de vídeo mostrados, EasyOCR es pésimo haciendo la lectura de las matrículas, al igual que Pytesseract. Por otro lado, SmolVLM, aunque en algunos casos su lectura no era del todo correcta, funcionaba mucho mejor que los otros dos.

<h2 align="center">Parte 4: Extras</h2>

En cuanto a los puntos extras, se han realizado dos de ellos; la anonimación de personas y el flujo de personas y vehículos.

Para la anonimación de personas, el modo de proseguir con este punto ha sido el usar una función desarrollada que aplica un blur a la detección de la persona o vehículo. Para esto, simplemente se aplica la función `cv2.GaussianBlur`.

Se presenta un fragmento del vídeo resultante:

<h3 align="center">Fragmento anonimizando personas y matrículas</h3> 

<div style="text-align: center;" align="center"> <img src="salida_anonimacin_de_personas_y_matriculas.gif"> </div>

En cuanto a la determinación del flujo, se ha definido un margen de varios píxeles en los bordes del vídeo para facilitar la detección de estos, pues se usan los boxes de detección de cada entidad para calcular hacia dónde se están moviendo y hacer el conteo de esta manera.

A continuación, se muestra un fragmento del vídeo resultado:

<h3 align="center">Fragmento determinando el flujo direccional</h3> 

<div style="text-align: center;" align="center"> <img src="salida_flujo_direccional.gif"> </div>

<h2 align="center">Parte 4b</h2>

<h3 align="center">Bibliografía</h3>

- [Repositorio usado como base y enunciado de esta práctica](https://github.com/otsedom/otsedom.github.io/tree/main/VC/P4)
