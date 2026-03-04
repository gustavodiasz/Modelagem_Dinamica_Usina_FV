import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

class MetodoDeWelch:
    def __init__(self, col_G='G'):
        self.col_G = col_G
        self.frec = None
        self.ampli_G = None
        self.ampli_P = None
        
    def AplicaMetodoDeWelch(self, df, coluna_potencia, win_size, NOME_COLUNA_PARA_PLOT='Informe O Nome Da Coluna Para O Plot', minuto_correção='minutos_correcao_potencia'):
        config_visual = {
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
            df = df.interpolate(method='linear')
            df = df.fillna(0) 
            
            g_pu = (df[self.col_G] / 1000).to_numpy()
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
            plt.figure(figsize=(10, 6))
            plt.loglog(f_g, Sgg, label='Irradiância (G)', alpha=0.8, color='orange')
            plt.loglog(f_p, Spp, label=NOME_COLUNA_PARA_PLOT, alpha=0.8, color='blue')
            
            plt.xlabel('Frequência [Hz]')
            plt.ylabel('PSD [PU²/Hz]')
            
            plt.title(f'Espectro de Potência - Janela de {int(win_size/(60*24))} dia(s).\n'
                      f'Irradiância (G) Vs {NOME_COLUNA_PARA_PLOT}\n'
                      f'Com Correção de falhas <= {minuto_correção} min.')
            
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
                
            plt.savefig(f'{pasta_saida}/ESPECTRO_{NOME_COLUNA_PARA_PLOT}.png', format='png', dpi=600, bbox_inches='tight')
            plt.show()
            plt.close()