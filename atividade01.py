import numpy as np

def funcao_softmax(valores, eixo=-1):
    valores = valores - np.max(valores, axis=eixo, keepdims=True)
    expoentes = np.exp(valores)
    return expoentes / np.sum(expoentes, axis=eixo, keepdims=True)

dimensao_modelo = 10
tamanho_sequencia = 10
dados_entrada = np.random.randn(1, tamanho_sequencia, dimensao_modelo)

pesos_q = np.random.randn(dimensao_modelo, dimensao_modelo)
pesos_k = np.random.randn(dimensao_modelo, dimensao_modelo)
pesos_v = np.random.randn(dimensao_modelo, dimensao_modelo)

query = np.matmul(dados_entrada, pesos_q)
chaves = np.matmul(dados_entrada, pesos_k)
valores = np.matmul(dados_entrada, pesos_v)

print(query.shape, chaves.shape, valores.shape)
