import pandas as pd
import numpy as np

class ObtemOGanhoDaUsinaMinutoAMinuto: 
    def __init__(self, area_hectares, col_p='P', col_g='G', col_zenith='Zenith'):
        """
        Determinar o vetor global de eficiência analítico (K).
        """
        self.area_hectares = area_hectares
        self.col_p = col_p
        self.col_g = col_g
        self.col_zenith = col_zenith
        fc = 0.021 * (self.area_hectares ** -0.5)
        self.tau = 1 / (2 * np.pi * fc)

    def gerar_vetor_k_global(self, df_entrada):
        """
        Varre TODA a série temporal reindexada para aprender e gerar 
        um Vetor K único e global usando a dedução analítica exata.
        """
        df_hist = df_entrada.copy()
        df_hist['P_t0'] = df_hist[self.col_p].shift(1)
        df_hist['G_t0'] = df_hist[self.col_g].shift(1)
        df_hist['G_medio'] = (df_hist[self.col_g] + df_hist['G_t0']) / 2
        
        mask_dia = (df_hist[self.col_zenith] <= 88) & \
                   (df_hist['G_medio'] > 0) & \
                   (df_hist[self.col_p] > 0)
        df_hist = df_hist[mask_dia].copy()

        # 3. Magnitude da Variação
        df_hist['delta_G'] = df_hist[self.col_g].diff().abs()
        var_diaria = df_hist.groupby(df_hist.index.date)['delta_G'].median()
        # Pega os 30% de dias mais "lisos" de toda a história da usina
        limite_var = var_diaria.quantile(0.30)
        dias_selecionados = var_diaria[var_diaria <= limite_var].index
        df_limpo = df_hist[np.isin(df_hist.index.date, dias_selecionados)].copy()        
        if df_limpo.empty:
            raise ValueError("Nenhum dia válido sobrou na série temporal após os filtros.")

        # 4. Equação Analítica Exata para K
        fator_exp = np.exp(-60.0 / self.tau)
        alfa_exato = 1 - fator_exp
        numerador = df_limpo[self.col_p] - (df_limpo['P_t0'] * fator_exp)
        denominador = alfa_exato * df_limpo['G_medio']
        df_limpo['K_inst'] = numerador / denominador
        # Trava contra outliers matemáticos de divisão
        df_limpo = df_limpo[(df_limpo['K_inst'] > 0) & (df_limpo['K_inst'] <= 2.0)]

        # 5. Agrupar por minuto do dia e extrair a mediana
        df_limpo['hora_minuto'] = df_limpo.index.strftime('%H:%M')
        vetor_k_global = df_limpo.groupby('hora_minuto')['K_inst'].median()
        return vetor_k_global

    def adicionar_coluna_k(self, df_entrada, vetor_k, nome_coluna='Ganho Da Usina (K)'):
        """
        Mapeia o vetor K gerado diretamente como uma nova coluna na série temporal.
        """
        df_out = df_entrada.copy()
        # Extrai a string HH:MM do índice para cruzar com o vetor
        hora_minuto = df_out.index.strftime('%H:%M')
        # Cria a coluna copiando o K ideal para cada linha correspondente
        df_out[nome_coluna] = hora_minuto.map(vetor_k)
        return df_out