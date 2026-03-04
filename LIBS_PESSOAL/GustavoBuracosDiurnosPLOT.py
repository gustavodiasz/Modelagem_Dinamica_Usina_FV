import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

class Analisador_De_Buracos_Diurno:
    def __init__(self):
        # Indentação corrigida para 4 espaços
        self.plot_style = {
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 12,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.figsize': (25, 45),
            'mathtext.fontset': 'stix'
        }

    def plotar_relatorio(self, df):
        # Filtra apenas o período diurno
        df_dia = df[df['Zenith'] < 88]
        
        # Calcula a diferença de tempo entre medições consecutivas
        diferencas = df_dia.index.to_series().diff()
        
        # Define o que é um buraco:
        # > 1 min: pulou pelo menos uma medição (assumindo freq 1 min)
        # < 720 min (12h): para evitar pegar a transição da noite (dia D para dia D+1)
        mask_buracos = (diferencas > pd.Timedelta('1min')) & (diferencas < pd.Timedelta('720min'))
        buracos_filtrados = diferencas[mask_buracos]

        print(f"Total de 'buracos' na amostragem: ... {len(buracos_filtrados)}")
        
        if len(buracos_filtrados) == 0:
            print("Nenhum gap encontrado.")
            return

        print(f"\nDiferença Máx: {buracos_filtrados.max()}\nDiferença Mín: {buracos_filtrados.min()}")

        # Separação por trimestres
        buracos_q1 = buracos_filtrados[buracos_filtrados.index.month <= 3]
        buracos_q2 = buracos_filtrados[(buracos_filtrados.index.month > 3) & (buracos_filtrados.index.month <= 6)]
        buracos_q3 = buracos_filtrados[(buracos_filtrados.index.month > 6) & (buracos_filtrados.index.month <= 9)]
        buracos_q4 = buracos_filtrados[buracos_filtrados.index.month > 9]
        
        configuracoes = [
            (buracos_q1, '1º Trimestre (Jan - Mar)'),
            (buracos_q2, '2º Trimestre (Abr - Jun)'),
            (buracos_q3, '3º Trimestre (Jul - Set)'),
            (buracos_q4, '4º Trimestre (Out - Dez)')
        ]

        with plt.rc_context(self.plot_style):
            fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(25, 30), sharey=True)
            fig.suptitle('Análise Geral de Falhas de Medição ao Longo do Ano: Períodos Diurnos\n Log-Linear', 
                         fontsize=35, fontweight='bold', y=0.97)
            
            for i, (dados, titulo) in enumerate(configuracoes):
                ax = axes[i]
                self._plotar_sub_periodo(ax, dados, titulo)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Garante que a pasta existe
            if not os.path.exists('./GRAFICOS'):
                os.makedirs('./GRAFICOS')

            nome_arquivo = './GRAFICOS/Analise_Geral_Falhas_Diurnas.pdf'
            plt.savefig(nome_arquivo, format='pdf', bbox_inches='tight')
            print(f"Gráfico salvo em: {nome_arquivo}")
            plt.show()

    def _plotar_sub_periodo(self, ax, dados, titulo):
        """Método auxiliar interno para plotar cada eixo"""
        fontsize_custom = 14
        
        if len(dados) == 0:
            ax.text(0.5, 0.5, "Sem falhas registradas neste trimestre", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=32, color='gray')
            ax.set_title(titulo, fontsize=fontsize_custom, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3)
            return

        minutos = dados.dt.total_seconds() / 60
        
        # CORREÇÃO CRÍTICA: width=0.02 (aprox 30 min visuais) ao invés de 0.5 (12 horas)
        # O eixo X é data, então 1 unidade = 1 dia.
        ax.bar(dados.index, minutos, width=0.02, color='red', alpha=0.5)
        
        ax.set_yscale('log')
        ax.set_title(titulo, fontsize=fontsize_custom, fontweight='bold')
        ax.set_ylabel('Duração (Min)', fontsize=fontsize_custom)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
        
        # Anotações
        for date, duration in zip(dados.index, minutos):
            ax.annotate(f'{int(duration)}', 
                        xy=(date, duration), 
                        xytext=(0, 3), textcoords='offset points', 
                        ha='center', fontsize=12, color='darkred')