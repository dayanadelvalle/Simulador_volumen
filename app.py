import streamlit as st
import pandas as pd
import io
from modelo import ModeloPacking

# ── Configuración de página ───────────────────────────────────
st.set_page_config(
    page_title="Simulador de Volumen de Packing",
    page_icon="📦",
    layout="centered",
)

st.title("📦 Simulador de Volumen de Packing")
st.markdown("Predice el volumen en m³ de tus grupos de despacho antes de empacar.")
st.divider()

# ── Estado de sesión ──────────────────────────────────────────
if 'modelo' not in st.session_state:
    st.session_state.modelo = ModeloPacking()

modelo: ModeloPacking = st.session_state.modelo

# ── PASO 1: Cargar histórico ──────────────────────────────────
st.subheader("Paso 1 — Cargar histórico de packing")

if not modelo.entrenado:
    st.info("El modelo aún no está entrenado. Sube el archivo histórico para comenzar.")

archivo_historico = st.file_uploader(
    "Selecciona el histórico (.xlsx)",
    type=["xlsx"],
    key="historico",
    help="Debe tener columnas: Grupo, Material, Cantidad entrega, Numero de caja, Caja, Cubicaje",
)

if archivo_historico:
    with st.spinner("Entrenando el modelo... esto puede tomar unos segundos ⏳"):
        try:
            df_hist = pd.read_excel(archivo_historico)
            registros, grupos, materiales = modelo.cargar_historico(df_hist)
            st.success(
                f"✅ Modelo entrenado con **{registros:,} registros** | "
                f"**{grupos:,} grupos** | **{materiales:,} materiales**"
            )
        except Exception as e:
            st.error(f"❌ Error al cargar el histórico: {e}")

st.divider()

# ── PASO 2: Predecir desde plantilla ─────────────────────────
st.subheader("Paso 2 — Subir plantilla de pedidos nuevos")

if not modelo.entrenado:
    st.warning("⚠️ Primero debes cargar el histórico en el Paso 1.")
else:
    st.markdown(
        "La plantilla debe tener estas columnas exactas en la **primera fila**: "
        "`Grupo` | `Material` | `Cantidad entrega`"
    )

    archivo_plantilla = st.file_uploader(
        "Selecciona la plantilla (.xlsx)",
        type=["xlsx"],
        key="plantilla",
    )

    if archivo_plantilla:
        with st.spinner("Calculando volúmenes estimados... 📐"):
            try:
                df_nuevos = pd.read_excel(archivo_plantilla)
                df_result = modelo.predecir_plantilla(df_nuevos)

                st.success(f"✅ Predicción lista para **{len(df_result) - 1} grupos**")
                st.divider()

                # ── Tabla de resultados ───────────────────────
                st.subheader("📊 Resultados por grupo")

                # Resaltar fila TOTAL
                def resaltar_total(row):
                    if row['Grupo'] == 'TOTAL GENERAL':
                        return ['background-color: #1f4e79; color: white; font-weight: bold'] * len(row)
                    return [''] * len(row)

                st.dataframe(
                    df_result.style.apply(resaltar_total, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )

                # ── Métricas resumen ──────────────────────────
                col1, col2, col3 = st.columns(3)
                total_vol = df_result[df_result['Grupo'] != 'TOTAL GENERAL']['Volumen estimado (m³)'].sum()
                num_grupos = len(df_result) - 1
                sin_hist   = df_result[
                    (df_result['Grupo'] != 'TOTAL GENERAL') &
                    (df_result['Sin historial'] != '✅ Todos encontrados')
                ].shape[0]

                col1.metric("Total grupos",        f"{num_grupos}")
                col2.metric("Volumen total (m³)",  f"{total_vol:.4f}")
                col3.metric("Grupos con materiales sin historial", f"{sin_hist}")

                # ── Descarga Excel ────────────────────────────
                st.divider()
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_result.to_excel(writer, index=False, sheet_name='Resultados')
                output.seek(0)

                st.download_button(
                    label="⬇️ Descargar resultados en Excel",
                    data=output,
                    file_name="resultado_volumen.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            except Exception as e:
                st.error(f"❌ Error al procesar la plantilla: {e}")

st.divider()
st.caption("Simulador de Volumen de Packing — modelo XGBoost entrenado con histórico de empaque del CEDI")
