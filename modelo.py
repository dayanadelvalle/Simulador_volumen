import pandas as pd
import numpy as np
from collections import defaultdict
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

COL_GRUPO     = 'Grupo'
COL_MATERIAL  = 'Material'
COL_CANTIDAD  = 'Cantidad entrega'
COL_NUM_CAJA  = 'Numero de caja'
COL_TIPO_CAJA = 'Caja'
COL_CUBICAJE  = 'Cubicaje'

FEATURES = [
    'num_materiales', 'cantidad_total', 'cantidad_max', 'cantidad_min',
    'num_tipos_caja', 'cubicaje_suma_naive', 'cubicaje_max',
    'cubicaje_min', 'cubicaje_prom', 'prop_sin_historial',
]

class ModeloPacking:
    def __init__(self):
        self.modelo_xgb         = None
        self.dict_modelo        = {}
        self.cubicaje_por_tipo  = {}
        self.cubicaje_promedio  = 0.069
        self.unidades_promedio  = 1
        self.factor_calibracion = 1.0
        self.entrenado          = False

    def cargar_historico(self, df_raw: pd.DataFrame):
        df_raw = df_raw.copy()
        df_raw.columns = df_raw.columns.str.strip()

        df_raw[COL_MATERIAL] = (
            df_raw[COL_MATERIAL].astype(str).str.strip()
            .str.replace(r'\.0$', '', regex=True)
        )
        df_raw[COL_CANTIDAD]  = pd.to_numeric(df_raw[COL_CANTIDAD].astype(str).str.replace(',', '.'), errors='coerce')
        df_raw[COL_CUBICAJE]  = pd.to_numeric(df_raw[COL_CUBICAJE].astype(str).str.replace(',', '.'), errors='coerce')
        df_raw[COL_NUM_CAJA]  = pd.to_numeric(df_raw[COL_NUM_CAJA], errors='coerce')
        df_raw[COL_TIPO_CAJA] = df_raw[COL_TIPO_CAJA].astype(str).str.strip()
        df_raw[COL_GRUPO]     = df_raw[COL_GRUPO].astype(str).str.strip()

        df = df_raw.dropna(subset=[COL_GRUPO, COL_MATERIAL, COL_NUM_CAJA, COL_CUBICAJE]).copy()
        df = df[df[COL_MATERIAL].str.len() > 3].copy()

        # Cubicaje mediano por tipo de caja
        self.cubicaje_por_tipo = (
            df.groupby(COL_TIPO_CAJA)[COL_CUBICAJE].median().to_dict()
        )

        # Target: volumen real por grupo
        df_cajas = (
            df.groupby([COL_GRUPO, COL_NUM_CAJA])
            .agg(
                tipo_caja=(COL_TIPO_CAJA, lambda x: x.mode()[0]),
                cubicaje =(COL_CUBICAJE,  lambda x: self.cubicaje_por_tipo.get(x.mode()[0], x.median())),
            )
            .reset_index()
        )
        target_grupo = (
            df_cajas.groupby(COL_GRUPO)['cubicaje'].sum()
            .reset_index().rename(columns={'cubicaje': 'volumen_real'})
        )

        # Modelo por material
        df_mat_caja = (
            df.groupby([COL_GRUPO, COL_NUM_CAJA, COL_MATERIAL])
            .agg(unidades=(COL_CANTIDAD, 'sum'))
            .reset_index()
            .merge(df_cajas[[COL_GRUPO, COL_NUM_CAJA, 'tipo_caja']], on=[COL_GRUPO, COL_NUM_CAJA])
        )
        modelo_material = (
            df_mat_caja.groupby(COL_MATERIAL)
            .agg(
                tipo_caja_frecuente=('tipo_caja', lambda x: x.mode()[0]),
                unidades_por_caja  =('unidades',  'median'),
                frecuencia         =('tipo_caja',  'count'),
            )
            .reset_index()
        )
        modelo_material['unidades_por_caja'] = modelo_material['unidades_por_caja'].clip(lower=1).round(0).astype(int)
        modelo_material['cubicaje'] = modelo_material['tipo_caja_frecuente'].map(self.cubicaje_por_tipo)

        mediana_global = np.median(list(self.cubicaje_por_tipo.values()))
        modelo_material['cubicaje'] = modelo_material['cubicaje'].fillna(mediana_global)

        self.dict_modelo       = modelo_material.set_index(COL_MATERIAL).to_dict(orient='index')
        self.cubicaje_promedio = modelo_material['cubicaje'].median()
        self.unidades_promedio = max(1, int(modelo_material['unidades_por_caja'].median()))

        # Dataset de entrenamiento
        registros = []
        for grupo_id, subdf in df.groupby(COL_GRUPO):
            mats  = subdf.groupby(COL_MATERIAL)[COL_CANTIDAD].sum().to_dict()
            feats = self._features_grupo(mats)
            feats[COL_GRUPO] = grupo_id
            registros.append(feats)

        df_features = pd.DataFrame(registros).merge(target_grupo, on=COL_GRUPO)

        X     = df_features[FEATURES]
        y_log = np.log1p(df_features['volumen_real'])

        X_train, _, y_train, _ = train_test_split(X, y_log, test_size=0.2, random_state=42)

        self.modelo_xgb = XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
        )
        self.modelo_xgb.fit(X_train, y_train)
        self.entrenado = True

        return len(df), df[COL_GRUPO].nunique(), len(self.dict_modelo)

    def _features_grupo(self, materiales_cantidades: dict) -> dict:
        cubicajes, cantidades, tipos_caja = [], [], set()
        sin_hist = 0
        cubicaje_suma_naive = 0.0

        for mat, cant in materiales_cantidades.items():
            m = str(mat).strip().replace('.0', '')
            if len(m) <= 3:
                continue
            cantidades.append(cant)
            if m in self.dict_modelo:
                info = self.dict_modelo[m]
                cub  = info['cubicaje']
                unid = max(1, info['unidades_por_caja'])
                tipo = info['tipo_caja_frecuente']
            else:
                cub  = self.cubicaje_promedio
                unid = self.unidades_promedio
                tipo = 'PROMEDIO'
                sin_hist += 1
            cubicajes.append(cub)
            tipos_caja.add(tipo)
            cubicaje_suma_naive += int(np.ceil(cant / unid)) * cub

        n = len(cantidades) if cantidades else 1
        return {
            'num_materiales'      : n,
            'cantidad_total'      : sum(cantidades),
            'cantidad_max'        : max(cantidades) if cantidades else 0,
            'cantidad_min'        : min(cantidades) if cantidades else 0,
            'num_tipos_caja'      : len(tipos_caja),
            'cubicaje_suma_naive' : cubicaje_suma_naive,
            'cubicaje_max'        : max(cubicajes) if cubicajes else self.cubicaje_promedio,
            'cubicaje_min'        : min(cubicajes) if cubicajes else self.cubicaje_promedio,
            'cubicaje_prom'       : np.mean(cubicajes) if cubicajes else self.cubicaje_promedio,
            'prop_sin_historial'  : sin_hist / n,
        }

    def predecir(self, materiales_cantidades: dict) -> dict:
        mats = {
            str(k).strip().replace('.0', ''): v
            for k, v in materiales_cantidades.items()
            if len(str(k).strip().replace('.0', '')) > 3
        }
        if not mats:
            return {'volumen_total_m3': 0.0, 'materiales_sin_historial': []}

        feats       = self._features_grupo(mats)
        X_pred      = pd.DataFrame([feats])[FEATURES]
        volumen_log = float(self.modelo_xgb.predict(X_pred)[0])
        volumen     = max(0.0, round(float(np.expm1(volumen_log)) * self.factor_calibracion, 4))
        sin_hist    = [m for m in mats if m not in self.dict_modelo]

        return {'volumen_total_m3': volumen, 'materiales_sin_historial': sin_hist}

    def predecir_plantilla(self, df_nuevos: pd.DataFrame) -> pd.DataFrame:
        df_nuevos = df_nuevos.copy()
        df_nuevos.columns = df_nuevos.columns.str.strip()
        df_nuevos['Material'] = (
            df_nuevos['Material'].astype(str).str.strip()
            .str.replace(r'\.0$', '', regex=True)
        )
        df_nuevos['Cantidad entrega'] = pd.to_numeric(
            df_nuevos['Cantidad entrega'].astype(str).str.replace(',', '.'), errors='coerce'
        ).fillna(0)
        df_nuevos['Grupo'] = df_nuevos['Grupo'].astype(str).str.strip()
        df_nuevos = df_nuevos[df_nuevos['Material'].str.len() > 3]

        resultados = []
        for grupo_id, subdf in df_nuevos.groupby('Grupo'):
            mats = subdf.groupby('Material')['Cantidad entrega'].sum().to_dict()
            res  = self.predecir(mats)
            resultados.append({
                'Grupo'                : grupo_id,
                'Volumen estimado (m³)': res['volumen_total_m3'],
                'Sin historial'        : ', '.join(res['materiales_sin_historial']) if res['materiales_sin_historial'] else '✅ Todos encontrados',
            })

        df_res = pd.DataFrame(resultados)
        total  = pd.DataFrame([{
            'Grupo'                : 'TOTAL GENERAL',
            'Volumen estimado (m³)': round(df_res['Volumen estimado (m³)'].sum(), 4),
            'Sin historial'        : '',
        }])
        return pd.concat([df_res, total], ignore_index=True)
