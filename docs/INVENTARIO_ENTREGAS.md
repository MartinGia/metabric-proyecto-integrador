# Inventario de entregas

Cada buzón del aula virtual admite **un solo archivo de hasta 100 MB**. Por eso, en cada entrega se sube un único PDF cuya portada contiene los enlaces a los archivos ejecutables (cuadernos en GitHub y Colab, código y dashboard). Todos los documentos siguen el formato APA 7 (estudiante) y tienen su versión en Word en la misma carpeta.

Carpeta de los documentos: `reports/entregas/`.

| Entrega | Actividad | Archivo que se sube al buzón | Ejecutables enlazados en la portada |
|---|---|---|---|
| Actividad 1 | Selección del dataset y planteamiento | `Actividad1_Seleccion_Dataset_METABRIC.pdf` | Repositorio y datos crudos |
| Entregable 1 (20 %) | Actividad 2: análisis exploratorio | `Informe_Entregable1_METABRIC.pdf` | `notebooks/01_EDA_Fase1.ipynb` (GitHub y Colab) |
| Actividad 3 | Limpieza e ingeniería de variables | `Actividad3_Limpieza_Ingenieria_Variables.pdf` | `notebooks/02_...`, `src/cleaning.py`, `src/features.py`, `src/preprocessing.py` |
| Entregable 2 (25 %) | Actividad 4: partición y modelo base | `Actividad4_Entregable2_Particion_Modelo_Base.pdf` | `notebooks/02_...` y `notebooks/03_...` (GitHub y Colab) |
| Actividad 5 | Modelado avanzado | `Actividad5_Modelado_Avanzado.pdf` | `notebooks/04_...`, `notebooks/05_...`, `models/` |
| Entregable 3 (25 %) | Actividad 6: evaluación e interpretabilidad | `Actividad6_Entregable3_Evaluacion_Interpretabilidad.pdf` | `notebooks/04_...`, `05_...` y `06_...` (GitHub y Colab) |
| Actividad 7 | Dashboard y narrativa | `Actividad7_Dashboard_Narrativa.pdf` | Dashboard en línea y `dashboard/app.py` |
| Actividad 8 | Informe técnico final | `Actividad8_Informe_Tecnico_Final.pdf` | Repositorio, dashboard y cuadernos |
| Entregable 4 (30 %) | Actividad 9: entrega final y sustentación | `Actividad9_Entrega_Final_Portada.pdf` | Repositorio, dashboard, informe final, presentación y cuadernos |

## Material de apoyo de la sustentación (no se sube al buzón)

| Archivo | Uso |
|---|---|
| `Actividad9_Presentacion_Sustentacion.pptx` | Presentación con el guion en las notas del orador (para ensayar y para el modo presentador) |
| `Actividad9_Presentacion_Sustentacion_sin_guion.pptx` | Misma presentación sin notas (para compartir) |
| `Actividad9_Guion_Sustentacion.docx` / `.pdf` | Guion completo por diapositiva con tiempos (unos 17 minutos) |

## Publicación antes de entregar

1. Crear el repositorio público `metabric-proyecto-integrador` en la cuenta `MartinGia` y subir esta carpeta (sin `docs/curso/`, que está excluida en `.gitignore`).
2. Comprobar que los botones «Open in Colab» del `README.md` abren cada cuaderno.
3. Desplegar el dashboard en Streamlit Community Cloud con el archivo principal `dashboard/app.py`, Python 3.12 y el subdominio `metabric-proyecto-integrador` (pasos en `dashboard/README.md`).
4. Abrir cada PDF y verificar que los enlaces de la portada responden.
5. Subir a cada buzón el PDF indicado en la tabla.
