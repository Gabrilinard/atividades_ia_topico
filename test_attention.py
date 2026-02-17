import numpy as np
from atividade01 import MecanismoAtencao

def executar_teste_simples():
    dimensao_modelo = 2
    tamanho_sequencia = 2

    dados_entrada = np.array([[[1.0, 2.0],
                               [3.0, 4.0]]])

    mecanismo = MecanismoAtencao(dimensao_modelo)

    mecanismo.pesos_consulta = np.zeros((dimensao_modelo, dimensao_modelo))
    mecanismo.pesos_chave = np.zeros((dimensao_modelo, dimensao_modelo))
    mecanismo.pesos_valor = np.eye(dimensao_modelo)

    pesos_esperados = np.array([[[0.5, 0.5],
                                 [0.5, 0.5]]])

    saida_esperada = np.array([[[2.0, 3.0],
                                [2.0, 3.0]]])

    saida, pesos_atencao = mecanismo.calcular_atencao(dados_entrada)
    print("Pesos de atenção calculados:\n", pesos_atencao)
    print("Pesos de atenção esperados:\n", pesos_esperados)
    print("Saída calculada:\n", saida)
    print("Saída esperada:\n", saida_esperada)

    assert pesos_atencao.shape == pesos_esperados.shape
    assert saida.shape == saida_esperada.shape
    assert np.allclose(pesos_atencao, pesos_esperados)
    assert np.allclose(saida, saida_esperada)
    print("Teste de atenção com exemplo numérico simples passou!!!")


if __name__ == "__main__":
    executar_teste_simples()

