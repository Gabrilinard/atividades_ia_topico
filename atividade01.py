import numpy as np

def funcao_softmax(valores, eixo=-1):
    valores = valores - np.max(valores, axis=eixo, keepdims=True)
    expoentes = np.exp(valores)
    return expoentes / np.sum(expoentes, axis=eixo, keepdims=True)

dimensao_modelo = 10
tamanho_sequencia = 10
dados_entrada = np.random.randn(1, tamanho_sequencia, dimensao_modelo)

print(funcao_softmax(dados_entrada, eixo=-1).shape)