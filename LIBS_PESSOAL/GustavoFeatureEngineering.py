import pandas as pd
import numpy as np
from pvlib.location import Location
from pvlib.irradiance import aoi

class FeatureEngineering:
    def preencher_noites_fisica(self, st_entrada, lat=-9.398611, lon=-40.50, angulo=88, angulo_painel = 15, fuso_horario='America/Recife'):

        """
            Converte o índice para Datetime, ajusta o fuso para 'America/Recife' 
            faz reamostragem (reindex) para frequência de 1 minuto
	    Adiciona Coluna Zenith (Ângulo Zênite Solar aparente).
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
    
        mask_noite = st['Zenith'] > angulo
        for col in ['G', 'P']:
            if col in st.columns:
                st.loc[mask_noite, col] = 0.0
        return st
        