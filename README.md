# 📦 Simulador de Volumen de Packing

Predice el volumen en m³ de grupos de despacho antes de empacar, usando un modelo XGBoost entrenado con el histórico de packing del CEDI.

---

## Archivos del proyecto

| Archivo | Descripción |
|---------|-------------|
| `app.py` | Interfaz Streamlit |
| `modelo.py` | Lógica del modelo XGBoost |
| `requirements.txt` | Dependencias Python |

---

## Cómo desplegar en Streamlit Community Cloud

1. Sube estos 3 archivos a un repositorio de GitHub (puede ser privado)
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Click en **New app**
4. Selecciona tu repositorio, rama `main` y archivo principal `app.py`
5. Click en **Deploy** — en 2-3 minutos tendrás el link para compartir

---

## Cómo usar la app

**Paso 1 — Cargar histórico**
Sube el archivo `HISTORICO_EMPAQUE.xlsx` con las columnas:
- `Grupo`
- `Material`
- `Cantidad entrega`
- `Numero de caja`
- `Caja`
- `Cubicaje`

El modelo se entrena automáticamente al subir el archivo (~10-30 segundos).

**Paso 2 — Subir plantilla**
Sube el archivo con los pedidos nuevos a predecir. Debe tener exactamente estas columnas en la primera fila:
- `Grupo`
- `Material`
- `Cantidad entrega`

La app muestra la tabla de resultados con el volumen estimado por grupo y permite descargar el Excel.

---

## Notas técnicas

- Materiales con código de 3 dígitos o menos se excluyen automáticamente (son servicios, no ocupan espacio físico)
- La caja tipo BULTO usa la mediana histórica de su cubicaje
- Materiales sin historial se imputan con la mediana global del modelo
- El modelo usa log-transform del target para manejar la distribución sesgada de volúmenes
