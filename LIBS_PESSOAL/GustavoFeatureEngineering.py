import pandas as pd
import numpy as np
from pvlib.location import Location
from pvlib.irradiance import aoi

class FeatureEngineering:
    def preencher_noites_fisica(self, st_entrada, lat=-9.398611, lon=-40.50, angulo=88, angulo_painel = 15, fuso_horario='America/Recife'):

        """
        Pipeline de pré-processamento para Machine Learning Solar (Physics-Informed).
        
        Realiza as seguintes operações:
        1.  **Normalização Temporal:** Converte o índice para Datetime, ajusta o fuso para 'America/Recife' 
            e faz reamostragem (reindex) para frequência de 1 minuto (preenchendo falhas com NaNs).
        
        2.  **Cálculo de Variáveis Astronômicas (Adiciona Colunas):**
            - 'Zenith': Ângulo Zênite Solar aparente.
            - 'Azimuth': Ângulo Azimute Solar.
            - 'AOI': Ângulo de Incidência (Angle of Incidence) considerando painéis inclinados a 15° Norte.
            
        3.  **Engenharia de Features (Transformações Trigonométricas):**
            - 'cos_Zenith': Cosseno do Zênite (Proxy para massa de ar).
            - 'cos_AOI': Cosseno do AOI (Proxy linear para irradiação efetiva).
            - 'sin_Azimuth' e 'cos_Azimuth': Decomposição vetorial para ciclicidade direcional.
            
        4.  **Limpeza de Ruído Noturno:**
            - Zera as colunas 'G' (Irradiação) e 'P' (Potência) quando 'Zenith' > 88°.
        
        Retorna:
            pd.DataFrame: DataFrame com índice contínuo e novas colunas de features físicas.
    	"""

        st = st_entrada.copy()
        st.index = pd.to_datetime(st.index)
    
        if st.index.tz is None:
            st.index = st.index.tz_localize(fuso_horario)
        else:
            st.index = st.index.tz_convert(fuso_horario)
    
        idx_completo = pd.date_range(st.index.min().floor('D'), st.index.max().ceil('D'), freq='1min', inclusive='left')
        st = st.reindex(idx_completo)
        st.index.name = 'Time'
        
        local = Location(lat, lon, tz=fuso_horario, name='Usina Solar Petrolina')
        solpos = local.get_solarposition(st.index)
        
        st['Zenith'] = solpos['apparent_zenith']
        st['Azimuth'] = solpos['azimuth']
    
        st['cos_Zenith'] = np.cos(np.radians(st['Zenith']))
        
        st['sin_Azimuth'] = np.sin(np.radians(st['Azimuth']))
        st['cos_Azimuth'] = np.cos(np.radians(st['Azimuth']))
        
        st['AOI'] = aoi(
            surface_tilt=angulo_painel,
            surface_azimuth=0,
            solar_zenith=st['Zenith'],
            solar_azimuth=st['Azimuth'])
        
        st['cos_AOI'] = np.cos(np.radians(st['AOI']))
    
        mask_noite = st['Zenith'] > angulo
        for col in ['G', 'P']:
            if col in st.columns:
                st.loc[mask_noite, col] = 0.0
        return st
        