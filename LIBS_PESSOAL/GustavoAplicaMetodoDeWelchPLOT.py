import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

class MetodoDeWelch:
    def __init__(self):
        self.col_G = None
        self.frec = None
        self.ampli_G = None
        self.ampli_P = None
        
    def AplicaMetodoDeWelch(self, df, coluna_potencia, coluna_iradiancia, win_size, NOME_COLUNA_DE_POTENCIA_PARA_PLOT='Informe O Nome Da Coluna De Potencia Para O Plot', 
NOME_COLUNA_DE_IRRADIANCIA_PARA_PLOT='Informe O Nome Da Coluna De Irradiancia Para O Plot',
minuto_correção='minutos_correcao_potencia'):
        config_visual = {
            'figure.figsize': (12, 6),
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 12,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'mathtext.fontset': 'stix'
        }
        
        with plt.rc_context(config_visual):
            
            g_pu = (df[coluna_iradiancia] / 1000).to_numpy()
            p_pu = (df[coluna_potencia] / 2500).to_numpy()
            
            # Parâmetros do Método de Welch
            fs = 1/60
            win_size = int(win_size)
            win = signal.windows.hann(win_size)
            noverlap = int(win_size / 2) 
            nfft = win_size 
            
            # Cálculo do Espectro
            f_g, Sgg = signal.csd(g_pu, g_pu, fs, window=win, noverlap=noverlap, nfft=nfft, scaling='density')
            f_p, Spp = signal.csd(p_pu, p_pu, fs, window=win, noverlap=noverlap, nfft=nfft, scaling='density') 
            
            # Guardando os resultados na instância
            self.frec = f_g
            self.ampli_G = Sgg
            self.ampli_P = Spp
        
            # Plotagem
            plt.loglog(f_g, Sgg, label=NOME_COLUNA_DE_IRRADIANCIA_PARA_PLOT, alpha=0.8, color='orange')
            plt.loglog(f_p, Spp, label=NOME_COLUNA_DE_POTENCIA_PARA_PLOT, alpha=0.8, color='blue')
            
            plt.xlabel('Frequência [Hz]')
            plt.ylabel('PSD [PU²/Hz]')
            
            # Formatação do Título Corrigida para Negrito e Caracteres Especiais
            nome_g_limpo = NOME_COLUNA_DE_IRRADIANCIA_PARA_PLOT.replace("_", r"\_")
            nome_p_limpo = NOME_COLUNA_DE_POTENCIA_PARA_PLOT.replace("_", r"\_")

            t1 = f'Espectro De Potência Determinado A Partir do Método de Welch Considerada Uma Janela de {int(win_size/(60*24))} dia(s).\n'
            t2 = r'Espectro do Sinal $\mathbf{' + nome_g_limpo + r'}$ Vs Espectro do Sinal $\mathbf{' + nome_p_limpo + r'}$' + '\n'
            t3 = rf'Espectros feito Após Correções Nas Falhas $\leq$ que {minuto_correção} Minutos Nos Dados de Potência $= 0$ Em período Diúrno.'
            
            plt.title(t1 + t2 + t3)
            
            # Frequências chave
            f_dia    = 1/(24*3600)
            f_hora   = 1/3600
            f_10min  = 1/600
            f_1min   = 1/60
            
            ticks_pos = [f_dia, f_hora, f_10min, f_1min]
            ticks_labels = ['1 Dia\n$~1.15e-5$', '1 Hora\n$~2.7e-4$', '10 Min\n$~1.6e-3$', '1 Min\n$~1.6e-2$']
            
            plt.xlim(1e-5, 2e-2)
            plt.xticks(ticks_pos, ticks_labels)
            plt.legend()
            plt.tight_layout()
            
            # Salvar
            pasta_saida = './GRAFICOS'
            if not os.path.exists(pasta_saida):
                os.makedirs(pasta_saida)
                
            plt.savefig(f'{pasta_saida}/ESPECTRO_{NOME_COLUNA_DE_IRRADIANCIA_PARA_PLOT}_Vs_{NOME_COLUNA_DE_POTENCIA_PARA_PLOT}.png', format='png', dpi=2*600, bbox_inches='tight')
            plt.show()
            plt.close()
