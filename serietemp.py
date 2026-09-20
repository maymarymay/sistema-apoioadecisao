import pandas as pd
import matplotlib.pyplot as plt

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    TEM_STATSMODELS = True
except ImportError:
    TEM_STATSMODELS = False

def carregar_dados(caminho_do_arquivo):
    dados = pd.read_csv(caminho_do_arquivo)
    dados['date'] = pd.to_datetime(dados['date'])
    dados = dados.sort_values('date').reset_index(drop=True)
    return dados


def tendmanual(serie, janela=7):
    return serie.rolling(window=janela, center=True).mean()


def sazmanual(serie, tendencia, periodo=7):
    sem_tendencia = serie - tendencia

    media_por_posicao = []
    for posicao in range(periodo):
        valores_da_posicao = sem_tendencia[posicao::periodo]
        media_por_posicao.append(valores_da_posicao.mean())

    sazonalidade = []
    for i in range(len(serie)):
        sazonalidade.append(media_por_posicao[i % periodo])

    return pd.Series(sazonalidade, index=serie.index)


def decompor_serie(dados, periodo=7):
    serie = dados.set_index('date')['meantemp'].asfreq('D')

    if TEM_STATSMODELS:
        resultado = seasonal_decompose(serie, model='additive', period=periodo)
        tendencia = resultado.trend
        sazonalidade = resultado.seasonal
        ruido = resultado.resid
    else:
        print('(statsmodels não encontrado, calculando a decomposição na mão)')
        tendencia = tendmanual(serie, janela=periodo)
        sazonalidade = sazmanual(serie, tendencia, periodo=periodo)
        ruido = serie - tendencia - sazonalidade

    return serie, tendencia, sazonalidade, ruido

def plot_decomposicao(serie, tendencia, sazonalidade, ruido, caminho_saida='decomposicao.png'):
    fig, eixos = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

    eixos[0].plot(serie.index, serie.values, color='tab:blue')
    eixos[0].set_title('Série original (temperatura média)')

    eixos[1].plot(tendencia.index, tendencia.values, color='tab:orange')
    eixos[1].set_title('Tendência')

    eixos[2].plot(sazonalidade.index, sazonalidade.values, color='tab:green')
    eixos[2].set_title('Sazonalidade')

    eixos[3].plot(ruido.index, ruido.values, color='tab:red')
    eixos[3].set_title('Ruído')

    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=110)
    plt.close(fig)


def distancia_euclidiana(serie1, serie2):
    valores1 = list(serie1)
    valores2 = list(serie2)

    soma_quadrados = 0
    for i in range(len(valores1)):
        diferenca = valores1[i] - valores2[i]
        soma_quadrados = soma_quadrados + diferenca ** 2

    return soma_quadrados ** 0.5


def distancia_manhattan(serie1, serie2):
    valores1 = list(serie1)
    valores2 = list(serie2)

    soma_diferencas = 0
    for i in range(len(valores1)):
        diferenca = abs(valores1[i] - valores2[i])
        soma_diferencas = soma_diferencas + diferenca

    return soma_diferencas

def dividir_em_partes(serie, numero_de_partes=11):
    tamanho_de_cada_parte = len(serie) // numero_de_partes

    partes = []
    inicio = 0
    for _ in range(numero_de_partes):
        fim = inicio + tamanho_de_cada_parte
        parte = serie.iloc[inicio:fim].reset_index(drop=True)
        partes.append(parte)
        inicio = fim

    return partes


def tres_mais_prox_dist(partes, funcao_distancia):
    subsequencia_atual = partes[-1]
    partes_anteriores = partes[:-1]

    distancias = []
    for indice, parte in enumerate(partes_anteriores):
        distancia = funcao_distancia(subsequencia_atual, parte)
        distancias.append((indice, distancia))

    distancias_ordenadas = sorted(distancias, key=lambda item: item[1])
    return distancias_ordenadas[:3]


def tres_mais_prox_cor(partes, metodo):
    subsequencia_atual = partes[-1]
    partes_anteriores = partes[:-1]

    correlacoes = []
    for indice, parte in enumerate(partes_anteriores):
        correlacao = subsequencia_atual.corr(parte, method=metodo)
        distancia_ate_um = abs(1 - correlacao)
        correlacoes.append((indice, correlacao, distancia_ate_um))

    correlacoes_ordenadas = sorted(correlacoes, key=lambda item: item[2])
    return correlacoes_ordenadas[:3]


def prox_e_ultimo_ponto(partes, indice):
    ultimo_ponto = partes[indice].iloc[-1]
    proximo_ponto = partes[indice + 1].iloc[0]
    return ultimo_ponto, proximo_ponto


def predicao_prox_pontos(partes, tres_mais_proximas):
    proximos_pontos = []
    for indice, correlacao, distancia in tres_mais_proximas:
        _, proximo_ponto = prox_e_ultimo_ponto(partes, indice)
        proximos_pontos.append(proximo_ponto)

    return sum(proximos_pontos) / len(proximos_pontos)


def predicao_da_distancia(partes, tres_mais_proximas, ultimo_ponto_atual):
    distancias = []
    for indice, correlacao, distancia in tres_mais_proximas:
        ultimo_ponto, proximo_ponto = prox_e_ultimo_ponto(partes, indice)
        distancias.append(proximo_ponto - ultimo_ponto)

    incremento = sum(distancias) / len(distancias)
    return ultimo_ponto_atual + incremento


def predicao_distancia_relativa(partes, tres_mais_proximas, ultimo_ponto_atual):
    distancias_relativas = []
    for indice, correlacao, distancia in tres_mais_proximas:
        ultimo_ponto, proximo_ponto = prox_e_ultimo_ponto(partes, indice)
        distancias_relativas.append((proximo_ponto - ultimo_ponto) / ultimo_ponto)

    incremento_relativo = sum(distancias_relativas) / len(distancias_relativas)
    return ultimo_ponto_atual * (1 + incremento_relativo)


def plotar_predicoes(serie, predicao1, predicao2, predicao3, caminho_saida='predicoes.png'):
    valores = list(serie.values)
    ultimos_pontos = valores[-20:] 
    eixo_x_real = range(1, len(ultimos_pontos) + 1)
    proximo_x = len(ultimos_pontos) + 1

    plt.figure(figsize=(9, 5))
    plt.plot(eixo_x_real, ultimos_pontos, marker='o', color='tab:blue', label='Série real (últimos dias)')

    plt.scatter([proximo_x], [predicao1], color='tab:orange', s=80, zorder=5,
                label='Predição - média dos próximos pontos')
    plt.scatter([proximo_x], [predicao2], color='tab:green', s=80, zorder=5,
                label='Predição - média da distância')
    plt.scatter([proximo_x], [predicao3], color='tab:red', s=80, zorder=5,
                label='Predição - média da distância relativa')

    plt.axvline(x=len(ultimos_pontos) + 0.5, color='gray', linestyle='--', linewidth=0.8)
    plt.title('Predição do próximo ponto da série (Pearson)')
    plt.xlabel('Dias)')
    plt.ylabel('Temperatura média')
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=110)
    plt.close()

if __name__ == '__main__':

    dados = carregar_dados('temperaturas.csv')

    serie, tendencia, sazonalidade, ruido = decompor_serie(dados, periodo=7)
    plot_decomposicao(serie, tendencia, sazonalidade, ruido)

    print('letra a)')
    print('gráfico em decomposicao.png' )
    print() 

    print('letra b)')
    serie1 = pd.Series([7, 16, 14, 25])
    serie2 = pd.Series([8, 13, 15, 22])
    print('Distância Euclidiana:', distancia_euclidiana(serie1, serie2))
    print('Distância de Manhattan:', distancia_manhattan(serie1, serie2))
    print()

    partes = dividir_em_partes(serie, numero_de_partes=11)
    print('letra c)')
    
    print('Versão 1: usando funções de distância')
    tres_euclidiana = tres_mais_prox_dist(partes, distancia_euclidiana)
    print('Euclidiana:', [(i, round(d, 2)) for i, d in tres_euclidiana])

    tres_manhattan = tres_mais_prox_dist(partes, distancia_manhattan)
    print('Manhattan:', [(i, round(d, 2)) for i, d in tres_manhattan])
    print()

    print('Versão 2: usando correlação')
    tres_pearson = tres_mais_prox_cor(partes, 'pearson')
    print('Pearson:', [(i, round(float(c), 3)) for i, c, _ in tres_pearson])

    tres_spearman = tres_mais_prox_cor(partes, 'spearman')
    print('Spearman:', [(i, round(float(c), 3)) for i, c, _ in tres_spearman])
    print()

    subsequencia_atual = partes[-1]
    ultimo_ponto_atual = subsequencia_atual.iloc[-1]

    predicao1 = predicao_prox_pontos(partes, tres_pearson)
    predicao2 = predicao_da_distancia(partes, tres_pearson, ultimo_ponto_atual)
    predicao3 = predicao_distancia_relativa(partes, tres_pearson, ultimo_ponto_atual)

    plotar_predicoes(serie, predicao1, predicao2, predicao3)

    print('letra d)')
    print('último ponto real da série:', round(float(ultimo_ponto_atual), 2))
    print('Predição com média dos próximos pontos:  ', round(float(predicao1), 2))
    print('Predição com média da distância:         ', round(float(predicao2), 2))
    print('Predição com média da distância relativa:', round(float(predicao3), 2))
    print('gráfico em predicoes.png')