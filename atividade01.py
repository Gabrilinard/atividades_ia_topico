import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def funcao_softmax(valores, eixo=-1):
    valores = valores - np.max(valores, axis=eixo, keepdims=True)
    expoentes = np.exp(valores)
    return expoentes / np.sum(expoentes, axis=eixo, keepdims=True)

class MecanismoAtencao:
    def __init__(self, dimensao_modelo):
        self.dimensao_modelo = dimensao_modelo
        self.pesos_consulta = np.random.randn(dimensao_modelo, dimensao_modelo)
        self.pesos_chave = np.random.randn(dimensao_modelo, dimensao_modelo)
        self.pesos_valor = np.random.randn(dimensao_modelo, dimensao_modelo)

    def calcular_atencao(self, entradas):
        consultas = np.matmul(entradas, self.pesos_consulta)
        chaves = np.matmul(entradas, self.pesos_chave)
        valores = np.matmul(entradas, self.pesos_valor)

        dimensao_chaves = consultas.shape[-1]
        scores = np.matmul(consultas, np.swapaxes(chaves, -2, -1))
        scores = scores / np.sqrt(dimensao_chaves)
        pesos_atencao = funcao_softmax(scores, eixo=-1)
        saida = np.matmul(pesos_atencao, valores)
        return saida, pesos_atencao

dimensao_modelo = 10
tamanho_sequencia = 10
dados_entrada = np.random.randn(1, tamanho_sequencia, dimensao_modelo)

mecanismo_atencao = MecanismoAtencao(dimensao_modelo)
representacao_vetorial, pesos_atencao = mecanismo_atencao.calcular_atencao(dados_entrada)

plt.figure(figsize=(10, 8))
sns.heatmap(pesos_atencao[0], annot=True, cmap='viridis')
plt.title("Heatmap de Pesos")
plt.xlabel("Posição das Chaves")
plt.ylabel("Posição das Consultas")
plt.show()
