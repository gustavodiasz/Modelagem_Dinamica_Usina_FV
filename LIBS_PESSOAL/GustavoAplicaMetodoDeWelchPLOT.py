from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

class MetodoDeWelch:

    def __init__(self):
        self.K_estatico = None
        self.f_corte = None

    def _get_config_visual(self):

        return {
            'figure.figsize': (12, 5),
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 12,
            'axes.labelsize': 12,
            'axes.titlesize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'mathtext.fontset': 'stix'}

    def _get_frequencias_chave(self):
        """Retorna as posições e rótulos do eixo X (Tempo -> Frequência)"""
        f_dia, f_hora, f_10min, f_1min = 1/(24*3600), 1/3600, 1/600, 1/60
        ticks_pos = [f_dia, f_hora, f_10min, f_1min]
        ticks_labels = ['1 Dia\n$~1.15e-5$', '1 Hora\n$~2.7e-4$', '10 Min\n$~1.6e-3$', '1 Min\n$~1.6e-2$']
        return ticks_pos, ticks_labels

    def plotar_psd(self, df, coluna_potencia, coluna_iradiancia, win_size,
                   NOME_COLUNA_DE_POTENCIA_PARA_PLOT='Potência', 
                   NOME_COLUNA_DE_IRRADIANCIA_PARA_PLOT='Irradiância',
                   minuto_correcao='0'):
        """Gera EXCLUSIVAMENTE o gráfico de Densidade Espectral de Potência (PSD)"""

        with plt.rc_context(self._get_config_visual()):
            g_pu = (df[coluna_iradiancia] / 1).to_numpy()
            p_pu = (df[coluna_potencia] / 1).to_numpy()

            fs = 1/60
            win_size = int(win_size)
            win = signal.windows.hann(win_size)
            noverlap = int(win_size / 2) 
            nfft = win_size 

            # Cálculo (Apenas Autoespectros)

            f_g, Sgg = signal.welch(g_pu, fs, window=win, noverlap=noverlap, nfft=nfft, scaling='density')
            f_p, Spp = signal.welch(p_pu, fs, window=win, noverlap=noverlap, nfft=nfft, scaling='density') 

            # Plotagem
            fig, ax = plt.subplots()
            nome_g_limpo = NOME_COLUNA_DE_IRRADIANCIA_PARA_PLOT.replace("_", r"\_")
            nome_p_limpo = NOME_COLUNA_DE_POTENCIA_PARA_PLOT.replace("_", r"\_")

            ax.loglog(f_g, Sgg, label=rf'PSD $\mathbf{{{nome_g_limpo}}}$', color='orange')
            ax.loglog(f_p, Spp, label=rf'PSD $\mathbf{{{nome_p_limpo}}}$', color='blue')

            ax.set_xlabel('Frequência [Hz]')
            ax.set_ylabel(r'PSD [$PU^2/Hz$]')

            t1 = f'Janela Welch: {int(win_size/(60*24))} dia(s) | Correção falhas $\\leq$ {minuto_correcao} min\n'
            ax.set_title(t1 + r'Autoespectros de Potência ($S_{PP}$) e Irradiância ($S_{GG}$)')

            ticks_pos, ticks_labels = self._get_frequencias_chave()
            ax.set_xlim(1e-6, 2e-2)
            ax.set_xticks(ticks_pos)
            ax.set_xticklabels(ticks_labels)
            ax.legend()
            plt.tight_layout()

            pasta_saida = './GRAFICOS'

            plt.savefig(f'{pasta_saida}/01_PSD_{NOME_COLUNA_DE_IRRADIANCIA_PARA_PLOT}_Vs_{NOME_COLUNA_DE_POTENCIA_PARA_PLOT}.png', format='png', dpi=2*600, bbox_inches='tight')

            plt.show()
            plt.close()

    def plotar_funcao_transferencia(self, df, coluna_potencia, coluna_iradiancia, win_size, minuto_correcao='[Sem Correção]'):
        """Gera EXCLUSIVAMENTE o diagrama de Bode |H(jw)| e determina K e fc"""
        with plt.rc_context(self._get_config_visual()):
            # Preparação dos Sinais (em P.U.)

            g_pu = (df[coluna_iradiancia] / 1000).to_numpy()
            p_pu = (df[coluna_potencia] / 2500).to_numpy()

            fs = 1/60
            win_size = int(win_size)
            win = signal.windows.hann(win_size)
            noverlap = int(win_size / 2) 
            nfft = win_size 

            # Cálculo para Função de Transferência (Sgg e Sgp)

            f_g, Sgg = signal.welch(g_pu, fs, window=win, noverlap=noverlap, nfft=nfft, scaling='density')
            f_gp, Sgp = signal.csd(g_pu, p_pu, fs, window=win, noverlap=noverlap, nfft=nfft, scaling='density')

            H_mag = np.zeros_like(Sgg)
            validos = Sgg > 0
            H_mag[validos] = np.abs(Sgp[validos] / Sgg[validos])
            K = H_mag[0]
            linha_corte_y = K / np.sqrt(2)

            # equação teórica da magnitude do filtro passa-baixas

            def filtro_teorico(f, fc):

                return K / np.sqrt(1 + (f / fc)**2)

            # Ajusta a curva usando os dados válidos (ignorando f=0)
            f_para_ajuste = f_g[1:]
            H_para_ajuste = H_mag[1:]

            try:
                # O curve_fit acha o fc que minimiza o erro entre a teoria e o seu ruido
                # p0=[0.01] é um 'chute' inicial

                popt, pcov = curve_fit(filtro_teorico, f_para_ajuste, H_para_ajuste, p0=[0.01])
                f_c = popt[0]
                H_teorico = filtro_teorico(f_para_ajuste, f_c)

            except:

                f_c = None
                H_teorico = None
                print("Aviso: O ajuste de curva falhou.")

            self.K_estatico = K
            self.f_corte = f_c


            # PLOTAGEM DO DIAGRAMA DE BODE

            fig, ax = plt.subplots()

            ax.loglog(f_g[1:], H_mag[1:], label=r'$|H(j\omega)|$ Empírico', color='darkblue', alpha=0.6, linewidth=1)

            if H_teorico is not None:
                ax.loglog(f_para_ajuste, H_teorico, label=r'Ajuste Teórico (Filtro 1ª Ordem)', color='cyan', linewidth=2.5)
            ax.axhline(y=K, color='red', linestyle='-', alpha=0.7, 
                       label=f'Ganho Estático ($K = {K:.3f}$)')
            ax.axhline(y=linha_corte_y, color='green', linestyle='--', alpha=0.8, 
                       label=r'Corte -3dB ($K/\sqrt{2}$' + f' $= {linha_corte_y:.3f}$)')

            if f_c is not None:
                ax.axvline(x=f_c, color='green', linestyle='--', alpha=0.8,
                           label=f'$f_c$ Ajustada $\\approx {f_c:.5f}$ Hz')
                ax.plot(f_c, linha_corte_y, 'go', markersize=6, zorder=5)

            t1 = f'Janela Welch: {int(win_size/(60*24))} dia(s) | Correção falhas $\\leq$ {minuto_correcao} min\n'
            ax.set_title(t1 + r'Identificação da Usina: Magnitude da Função de Transferência $|H(j\omega)|$')
            ax.set_xlabel('Frequência [Hz]')
            ax.set_ylabel('Magnitude [PU/PU]')

            ticks_pos, ticks_labels = self._get_frequencias_chave()
            ax.set_xlim(1e-5, 2e-2)
            ax.set_xticks(ticks_pos)
            ax.set_xticklabels(ticks_labels)
            ax.legend(loc='lower left')
            ax.grid(True, which="both", ls="--", alpha=0.5)   

            plt.tight_layout()
            pasta_saida = './GRAFICOS'

            os.makedirs(pasta_saida, exist_ok=True)
            plt.savefig(f'{pasta_saida}/02_FUNCAO_TRANSFERENCIA_H_jw.png', format='png', dpi=1200, bbox_inches='tight')
            plt.show()
            plt.close()