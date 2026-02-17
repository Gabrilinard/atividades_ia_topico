import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def funcao_softmax(valores, eixo=-1):
    valores = valores - np.max(valores, axis=eixo, keepdims=True)
    expoentes = np.exp(valores)
    return expoentes / np.sum(expoentes, axis=eixo, keepdims=True)

def atencao_produto_escalar_escalonado(entradas, pesos_consulta, pesos_chave, pesos_valor):
    consultas = np.matmul(entradas, pesos_consulta)
    chaves = np.matmul(entradas, pesos_chave)
    valores = np.matmul(entradas, pesos_valor)

    dimensao_chaves = consultas.shape[-1]
    scores = np.matmul(consultas, np.swapaxes(chaves, -2, -1))
    scores = scores / np.sqrt(dimensao_chaves)
    pesos_atencao = funcao_softmax(scores, eixo=-1)
    saida = np.matmul(pesos_atencao, valores)
    return saida, pesos_atencao

dimensao_modelo = 10
tamanho_sequencia = 10
dados_entrada = np.random.randn(1, tamanho_sequencia, dimensao_modelo)

pesos_consulta = np.random.randn(dimensao_modelo, dimensao_modelo)
pesos_chave = np.random.randn(dimensao_modelo, dimensao_modelo)
pesos_valor = np.random.randn(dimensao_modelo, dimensao_modelo)

representacao_vetorial, pesos_atencao = atencao_produto_escalar_escalonado(
    dados_entrada, pesos_consulta, pesos_chave, pesos_valor
)

plt.figure(figsize=(10, 8))
sns.heatmap(pesos_atencao[0], annot=True, cmap='viridis')
plt.title("Heatmap de Pesos")
plt.xlabel("Posição das Chaves")
plt.ylabel("Posição das Consultas")
plt.show()