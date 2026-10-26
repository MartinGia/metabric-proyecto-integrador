# Dashboard interactivo (Fase 4, actividad 7)

Aplicación en Streamlit que comunica los resultados del proyecto con cinco pestañas:

| Pestaña | Contenido | Módulo exigido por la guía |
|---|---|---|
| Resumen | Tarjetas KPI de la cohorte filtrada y de los modelos, narrativa en tres ideas, curvas de supervivencia de Kaplan-Meier | Tarjetas KPI |
| Exploración | Diagrama de dispersión configurable (variables clínicas o genes) | Visualización |
| Predicción en vivo | Perfil editable (variables clínicas y 10 genes clave) con NPI estimado (Ridge), probabilidades de subtipo PAM50 y clúster asignado | Módulo de predicción |
| Segmentación | Clústeres K-Means en el espacio PCA (2D o 3D), perfiles promedio y composición por subtipo | Módulo de clustering |
| Modelos y explicabilidad | Métricas en prueba con IC 95 %, cumplimiento SMART y variables más importantes (SHAP) | Comunicación de resultados |

La barra lateral contiene los **filtros dinámicos**: rangos de tiempo de seguimiento, edad y NPI; subtipo, grupo por receptores, clúster, grado y subconjunto. METABRIC no trae fechas de calendario, así que el rango temporal usa los meses de seguimiento desde el diagnóstico.

## Ejecución local

Desde la raíz del repositorio:

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

## Despliegue en Streamlit Community Cloud

1. Publicar el repositorio en GitHub (público).
2. Entrar a https://share.streamlit.io con la cuenta de GitHub y elegir **Create app**.
3. Seleccionar el repositorio, la rama `main` y el archivo principal `dashboard/app.py`.
4. En **Advanced settings**, elegir **Python 3.12**.
5. Pulsar **Deploy**. La aplicación usa los modelos de `models/` y los datos de `data/processed/`, que deben estar en el repositorio.

La aplicación es académica y no debe usarse para decisiones clínicas.
