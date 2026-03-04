import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Analisador_De_Potencia_Zero_Com_Sol_Presente:
    def __init__(self, col_p='P', col_g='G', limite_g=0):
        """
        Inicializa o analisador com os nomes das colunas e parâmetros padrão.
        
        :param col_p: Nome da coluna de Potência.
        :param col_g: Nome da coluna de Irradiação.
        :param limite_g: Valor mínimo de irradiação para considerar sol presente.
        """
        self.col_p = col_p
        self.col_g = col_g
        self.limite_g = limite_g
        self.config_visual = {
                            'font.family': 'serif',
                            'font.serif': ['Times New Roman'],
                            'font.size': 12,
                            'axes.labelsize': 12,
                            'axes.titlesize': 14,
                            'xtick.labelsize': 10,
                            'ytick.labelsize': 10,
                            'legend.fontsize': 10,
                            'figure.figsize': (25, 45),
                            'mathtext.fontset': 'stix'}
        
    def processar_eventos(self, df):
        """
        Método interno para calcular os eventos de falha antes de plotar.
        Retorna uma Series com as durações dos eventos.
        """
        if self.col_p not in df.columns or self.col_g not in df.columns:
            raise ValueError(f"Colunas '{self.col_p}' ou '{self.col_g}' não encontradas.")
        mask_zero_power = (df[self.col_p].values <= 0) & (df[self.col_g].values > self.limite_g)
        if not mask_zero_power.any():
            return pd.Series(dtype=float)
        series_mask = pd.Series(mask_zero_power, index=df.index)
        blocos_ids = (series_mask != series_mask.shift()).cumsum()
        df_falhas = df.loc[series_mask]
        ids_falhas = blocos_ids[series_mask]
        grouped = df_falhas.index.to_series().groupby(ids_falhas)
        resumo = grouped.agg(['min', 'max'])
        delta_tempo = (resumo['max'] - resumo['min']).dt.total_seconds() / 60
        duracao = delta_tempo + 1
        return pd.Series(data=duracao.values, index=resumo['min'])

    def plotar_relatorio(self, df, figsize=(25, 45), salvar_pdf=True, nome_arquivo='./GRAFICOS/Potencia_Zero_Com_Sol_Pesente.pdf'):
        """
        Gera o gráfico de análise bimestral de indisponibilidade.
        """
        gaps_day = self.processar_eventos(df)
        if gaps_day.empty:
            
            print("Nenhum evento de indisponibilidade com sol presente foi detectado.")
            
            return
        with plt.rc_context(self.config_visual):
            fig, axes = plt.subplots(nrows=6, ncols=1, figsize=figsize, sharey=True)
            fig.suptitle(f'Análise de Indisponibilidade de Geração\n($P=0$ com $G > {self.limite_g}$)\n Distribuição Bimestral', 
                         fontsize=35, fontweight='bold', y=0.98)
            config_bimestres = [
                (0, [1, 2], '1º Bimestre (Jan - Fev)'),
                (1, [3, 4], '2º Bimestre (Mar - Abr)'),
                (2, [5, 6], '3º Bimestre (Mai - Jun)'),
                (3, [7, 8], '4º Bimestre (Jul - Ago)'),
                (4, [9, 10], '5º Bimestre (Set - Out)'),
                (5, [11, 12], '6º Bimestre (Nov - Dez)')]
            for i, meses, titulo in config_bimestres:
                ax = axes[i]
                dados_bimestre = gaps_day[gaps_day.index.month.isin(meses)]
                if len(dados_bimestre) == 0:
                    ax.text(0.5, 0.5, "Sem registros neste bimestre", 
                            ha='center', va='center', transform=ax.transAxes, fontsize=18, color='gray')
                    ax.set_title(titulo, fontsize=24, fontweight='bold')
                    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                else:
                    ax.bar(dados_bimestre.index, dados_bimestre.values, width=0.2, color='blue', edgecolor='darkblue', alpha=0.6)
                    ax.set_yscale('log')
                    ax.set_title(titulo, fontsize=24, fontweight='bold')
                    ax.set_ylabel('Duração (Min)', fontsize=18)
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
                    ax.tick_params(axis='x', labelsize=18, rotation=0)
                    ax.tick_params(axis='y', labelsize=18)
                    for date, duration in zip(dados_bimestre.index, dados_bimestre.values):
                        ax.annotate(f'{int(duration)}', 
                                    xy=(date, duration), 
                                    xytext=(0, 5), textcoords='offset points', 
                                    ha='center', fontsize=12, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            if salvar_pdf:
                plt.savefig(nome_arquivo, format='pdf', bbox_inches='tight')
                
                print(f"Gráfico salvo como: {nome_arquivo}")

            plt.show()