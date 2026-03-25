import pandas as pd
import numpy as np

class ReconstrutorUsinaSolar:
    def __init__(self, area_hectares=1.48, limite_minutos=15, janela_vizinhos=10, col_p='P', col_g='G'):
        """
        Classe para correção de falhas em dados de usinas solares usando filtro Low-Pass.
        
        Parâmetros:
        :param area_hectares: Área total da usina (S) para cálculo da inércia
        :param limite_minutos: Tamanho máximo da falha a ser corrigida.
        :param janela_vizinhos: Quantos minutos antes/depois usar para estimar a eficiência (K).
        """
        self.area_hectares = area_hectares
        self.limite_minutos = limite_minutos
        self.janela_vizinhos = janela_vizinhos
        self.col_p = col_p
        self.col_g = col_g
        
        fc = 0.021 * (self.area_hectares ** -0.5)
        self.tau = 1 / (2 * np.pi * fc)

    def _estimar_k_local(self, df, idx_inicio, idx_fim):
        """
        Calcula a eficiência (K) local usando a MEDIANA dos vizinhos.
        Isso define a relação P/G esperada para aquele momento do dia.
        """
        pos_inicio = df.index.get_loc(idx_inicio)
        pos_fim = df.index.get_loc(idx_fim)
        
        inicio_viz = max(0, pos_inicio - self.janela_vizinhos)
        fim_viz = min(len(df), pos_fim + self.janela_vizinhos + 1)
        
        df_vizinhos = df.iloc[inicio_viz : fim_viz]
        
        mask_validos = (df_vizinhos[self.col_p] > 0) & (df_vizinhos[self.col_g] > 0)
        vizinhos_bons = df_vizinhos[mask_validos]
        
        if len(vizinhos_bons) >= 2:
            return (vizinhos_bons[self.col_p] / vizinhos_bons[self.col_g]).median()
        return None

    def _preparar_dataframe(self, df, nome_coluna_saida):
        df_out = df.sort_index().copy()
        
        if not isinstance(df_out.index, pd.DatetimeIndex):
             try:
                df_out.index = pd.to_datetime(df_out.index)
             except:
                raise ValueError("O índice precisa ser Datetime para cálculo físico do Delta T.")
        
        if nome_coluna_saida not in df_out.columns:
            df_out[nome_coluna_saida] = df_out[self.col_p].copy()
            
        mask_falha = (df_out[self.col_p] <= 0) & (df_out[self.col_g] > 0)
        # Cria IDs únicos para cada bloco consecutivo de falha
        blocos = (mask_falha != mask_falha.shift()).cumsum()
        
        return df_out, mask_falha, blocos

    # =========================================================================
    # MÉTODO 1: EULER BACKWARD
    # =========================================================================
        
    def reconstruir_via_euler(self, df):
        """
        Aplica a correção usando a discretização de Euler.
        Fórmula: P[t] = alpha * (K*G) + (1-alpha) * P[t-1]
        """
        NOME_COLUNA = 'P_Corrigido_Por_Euler'
        df_out, mask_falha, blocos = self._preparar_dataframe(df, NOME_COLUNA)
        grupos_falha = df_out[mask_falha].groupby(blocos)
        count = 0

        for _, dados_bloco in grupos_falha:
            if len(dados_bloco) > self.limite_minutos: continue

            # 1. Estimar K Local
            K_local = self._estimar_k_local(df_out, dados_bloco.index[0], dados_bloco.index[-1])
            if K_local is None: continue

            # 2. Configurar estado inicial
            pos_inicio = df_out.index.get_loc(dados_bloco.index[0])
            if pos_inicio > 0:
                idx_anterior = df_out.index[pos_inicio - 1]
                p_anterior = df_out.loc[idx_anterior, NOME_COLUNA]
                t_anterior = idx_anterior
                if pd.isna(p_anterior) or p_anterior <= 0: 
                    p_anterior = K_local * df_out.loc[idx_anterior, self.col_g]
            else:
                continue 

            # 3. Loop de Correção
            for t_atual in dados_bloco.index:
                delta_t = (t_atual - t_anterior).total_seconds()
                if delta_t <= 0: delta_t = 60.0 
                
                # alpha = dt / (tau + dt)
                alpha = delta_t / (self.tau + delta_t)
                g_atual = df_out.loc[t_atual, self.col_g]

                # Aplicação da Fórmula
                p_novo = (alpha * (K_local * g_atual)) + ((1 - alpha) * p_anterior)
                
                df_out.loc[t_atual, NOME_COLUNA] = p_novo
                p_anterior = p_novo
                t_anterior = t_atual
        
            count += 1
            
        return df_out

    # =========================================================================
    # MÉTODO 2: ANALÍTICO com IRRADIAÇÃO CONSTANTE
    # =========================================================================
    def reconstruir_via_analitico(self, df):
        """
        Aplica a correção usando a Solução Analítica Exata (Irradiação constante no intervalo).
        Fórmula: P(t) = P(t-1)*e^(-dt/tau) + K*G(t)*(1 - e^(-dt/tau))
        """
        NOME_COLUNA = 'P_Corrigido_Por_Integral_Analítica'
        df_out, mask_falha, blocos = self._preparar_dataframe(df, NOME_COLUNA)
        grupos_falha = df_out[mask_falha].groupby(blocos)
        count = 0

        for _, dados_bloco in grupos_falha:
            if len(dados_bloco) > self.limite_minutos: continue

            K_local = self._estimar_k_local(df_out, dados_bloco.index[0], dados_bloco.index[-1])
            if K_local is None: continue

            pos_inicio = df_out.index.get_loc(dados_bloco.index[0])
            if pos_inicio > 0:
                idx_anterior = df_out.index[pos_inicio - 1]
                p_anterior = df_out.loc[idx_anterior, NOME_COLUNA]
                t_anterior = idx_anterior
                if pd.isna(p_anterior) or p_anterior <= 0:
                    p_anterior = K_local * df_out.loc[idx_anterior, self.col_g]
            else:
                continue

            for t_atual in dados_bloco.index:
                delta_t = (t_atual - t_anterior).total_seconds()
                if delta_t <= 0: delta_t = 60.0
                
                g_atual = df_out.loc[t_atual, self.col_g]
                
                # --- FÓRMULA ANALÍTICA --- 
                fator_decaimento = np.exp(-delta_t / self.tau)
                alpha_analitico = 1 - fator_decaimento
                
                p_novo = (p_anterior * fator_decaimento) + ((K_local * g_atual) * alpha_analitico)

                df_out.loc[t_atual, NOME_COLUNA] = p_novo
                p_anterior = p_novo
                t_anterior = t_atual
            
            count += 1
        return df_out

    # =========================================================================
    # MÉTODO 3: INTEGRAL MIDPOINT
    # =========================================================================
    
    def reconstruir_via_integral_midpoint(self, df):
        """
        Aplica a correção baseada na aproximação da integral pelo ponto médio (t*).
        Assume que G varia linearmente entre t-1 e t
        Fórmula: P(t) = P(t-1)*e^(-dt/tau) + (K/tau * dt * G_medio * e^(-dt/2tau))
        """
        NOME_COLUNA = 'P_Corrigido_Por_G_Constante_Na_Falha'
        df_out, mask_falha, blocos = self._preparar_dataframe(df, NOME_COLUNA)
        grupos_falha = df_out[mask_falha].groupby(blocos)
        count = 0

        for _, dados_bloco in grupos_falha:
            if len(dados_bloco) > self.limite_minutos: continue

            K_local = self._estimar_k_local(df_out, dados_bloco.index[0], dados_bloco.index[-1])
            if K_local is None: continue

            pos_inicio = df_out.index.get_loc(dados_bloco.index[0])
            if pos_inicio > 0:
                idx_anterior = df_out.index[pos_inicio - 1]
                p_anterior = df_out.loc[idx_anterior, NOME_COLUNA]
                g_anterior = df_out.loc[idx_anterior, self.col_g] # G anterior para média
                t_anterior = idx_anterior
                if pd.isna(p_anterior) or p_anterior <= 0:
                    p_anterior = K_local * df_out.loc[idx_anterior, self.col_g]
            else:
                continue

            for t_atual in dados_bloco.index:
                delta_t = (t_atual - t_anterior).total_seconds()
                if delta_t <= 0: delta_t = 60.0
                
                g_atual = df_out.loc[t_atual, self.col_g]
                
                # Cálculo da média da irradiação no intervalo
                g_medio = (g_atual + g_anterior) / 2
                
                # --- APLICAÇÃO DA FÓRMULA ---
                termo_inercia = p_anterior * np.exp(-delta_t / self.tau)
                
                # Termo da integral aproximada no ponto médio t*
                fator_exponencial_meio = np.exp(-delta_t / (2 * self.tau))
                termo_entrada = (K_local / self.tau) * fator_exponencial_meio * g_medio * delta_t
                
                p_novo = termo_inercia + termo_entrada

                df_out.loc[t_atual, NOME_COLUNA] = p_novo
                
                # Atualiza variáveis (G atual vira G anterior para o próximo passo)
                p_anterior = p_novo
                g_anterior = g_atual 
                t_anterior = t_atual
            
            count += 1

        return df_out

    # =========================================================================
    # MÉTODO 4: ANALÍTICO com VETOR K (ATUALIZADO PARA IGNORAR G_ANTERIOR = NaN)
    # =========================================================================
    
    def reconstruir_potencia_integral_analitica(self, df, col_k='Ganho Da Usina (K)', nome_col_saida='P_Corrigido'):
        """
        Reconstrói as falhas de Potência propagando a inércia da usina através da
        solução exata da equação diferencial (Analítica).
        Preserva a coluna original e retorna o DataFrame com a nova coluna corrigida.
        """
        df_out = df.copy()
        df_out[nome_col_saida] = df_out[self.col_p].copy()
        
        fator_exp = np.exp(-60.0 / self.tau)
        alfa_exato = 1 - fator_exp
    
        mask_falha = (df_out[self.col_p] == 0) & (df_out[self.col_g] > 0)
        
        if not mask_falha.any():
            print("Nenhuma falha para reconstruir encontrada.")
            return df_out
    
        blocos_falha = (mask_falha != mask_falha.shift()).cumsum()
        grupos_de_falha = df_out[mask_falha].groupby(blocos_falha)
    
        for _, dados_bloco in grupos_de_falha:
            
            idx_inicio = dados_bloco.index[0]
            pos_inicio = df_out.index.get_loc(idx_inicio)
            
            if pos_inicio == 0: continue # Se a falha for na linha 0, não tem passado para usar
    
            p_anterior = df_out.iloc[pos_inicio - 1][nome_col_saida]
            g_anterior = df_out.iloc[pos_inicio - 1][self.col_g]
    
            for t_atual in dados_bloco.index:
                g_atual = df_out.loc[t_atual, self.col_g]
                k_atual = df_out.loc[t_atual, col_k]
                
                if pd.isna(k_atual): continue
                
                # Ajusta o G médio: se não há passado válido, a média é o presente
                if pd.isna(g_anterior):
                    g_medio = g_atual
                else:
                    g_medio = (g_atual + g_anterior) / 2
                    
                # Ajusta o P inicial: se P_anterior é inválido, cria um novo estado 
                # baseado no G que estiver disponível (evitando multiplicar K por NaN)
                if pd.isna(p_anterior) or p_anterior <= 0:
                    g_referencia = g_atual if pd.isna(g_anterior) else g_anterior
                    p_anterior = k_atual * g_referencia
                    
                # APLICAÇÃO DA RESOLUCAO ANALÍTICA
                termo_inercia = p_anterior * fator_exp
                termo_entrada = k_atual * g_medio * alfa_exato
                p_novo = termo_inercia + termo_entrada
                
                df_out.loc[t_atual, nome_col_saida] = p_novo
                
                # Atualiza para o próximo loop (neste ponto, p_novo e g_atual NUNCA são NaNs)
                p_anterior = p_novo
                g_anterior = g_atual
    
        return df_out