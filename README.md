# Laboratório 01 – Scaled Dot-Product Attention (from scratch)

Este repositório contém a implementação manual do mecanismo de **Scaled Dot-Product Attention**, seguindo o enunciado do laboratório:

- Biblioteca: NumPy (sem uso de camadas prontas de deep learning)
- Modelagem: geração de Q, K, V via multiplicação por matrizes de pesos
- Algoritmo: implementação manual de `ATTENTION(Q, K, V)`
- Análise: visualização dos pesos de atenção com heatmap
- Teste: script separado com exemplo numérico simples

---

## 1. Como rodar o código

### 1.1. Instalar dependências

Requisitos principais:

- Python 3.x
- NumPy
- Matplotlib
- Seaborn

Instalação (via `pip`):

```bash
pip install numpy matplotlib seaborn
```

### 1.2. Rodar o código principal (gerar o heatmap de atenção)

Na pasta do projeto (`Topicos em IA`), execute:

```bash
python atividade01.py
```

O script irá:

- Gerar uma sequência aleatória de vetores de dimensão `dimensao_modelo`.
- Aplicar o mecanismo de atenção (`MecanismoAtencao`).
- Exibir um **heatmap** com os pesos de atenção entre as posições da sequência.

### 1.3. Rodar o script de teste

Para validar a implementação com um exemplo numérico simples, execute:

```bash
python test_attention.py
```

O teste:

- Monta um exemplo pequeno (dimensão 2, sequência de 2 elementos).
- Configura pesos especiais para Q, K e V.
- Compara a saída da classe `MecanismoAtencao` com o resultado esperado.
- Imprime a mensagem `Teste de atenção com exemplo numérico simples passou!!!` se tudo estiver correto.

---

## 2. Normalização por 1 / sqrt(d_k)

A equação de referência do Scaled Dot-Product Attention pode ser escrita em texto como:

`ATTENTION(Q, K, V) = softmax((Q K^T) / sqrt(d_k)) * V`

No código, isso aparece dentro do método `calcular_atencao` da classe `MecanismoAtencao`:

- `consultas` (Q), `chaves` (K) e `valores` (V) são obtidos multiplicando a entrada pelas matrizes de pesos:
  - `consultas = entradas @ pesos_consulta`
  - `chaves = entradas @ pesos_chave`
  - `valores = entradas @ pesos_valor`
- Calculamos os escores `QK^T`:
  - `scores = np.matmul(consultas, np.swapaxes(chaves, -2, -1))`
- Aplicamos a **normalização pela raiz da dimensão das chaves** (`d_k`):

```python
dimensao_chaves = consultas.shape[-1]
scores = scores / np.sqrt(dimensao_chaves)
```

Aqui, `dimensao_chaves` é exatamente `d_k`. A divisão por `np.sqrt(dimensao_chaves)` implementa o fator `1 / sqrt(d_k)`.

Em seguida, aplicamos o softmax:

```python
pesos_atencao = funcao_softmax(scores, eixo=-1)
```

Essa normalização evita que os valores de `scores` fiquem muito grandes quando `d_k` cresce, o que poderia saturar o softmax e prejudicar o gradiente (como discutido no paper *“Attention Is All You Need”*).

---

## 3. Exemplo de input e output esperado (teste numérico)

O arquivo `test_attention.py` contém um exemplo numérico simples que pode ser descrito assim:

### 3.1. Entrada

- Dimensão do modelo: `dimensao_modelo = 2`
- Sequência com 2 vetores:

  `X = [[1, 2], [3, 4]]`

No código, isso é representado como um batch de tamanho 1:

```python
dados_entrada = np.array([[[1.0, 2.0],
                           [3.0, 4.0]]])
```

Os pesos da atenção são configurados manualmente no teste:

```python
mecanismo.pesos_consulta = np.zeros((2, 2))   # Q = 0
mecanismo.pesos_chave = np.zeros((2, 2))      # K = 0
mecanismo.pesos_valor = np.eye(2)             # V = X
```

Com isso:

- Q = 0 e K = 0 (logo, QK^T = 0) para todas as posições.
- Softmax dos escores iguais resulta em **pesos uniformes**.

### 3.2. Pesos de atenção esperados

Os pesos de atenção resultantes são:

`Pesos = [[0.5, 0.5], [0.5, 0.5]]`

No código:

```python
pesos_esperados = np.array([[[0.5, 0.5],
                             [0.5, 0.5]]])
```

### 3.3. Saída esperada

Como os pesos são uniformes, cada posição da saída é a **média** dos dois vetores de entrada:

`Saída = [[2, 3], [2, 3]]`

No código:

```python
saida_esperada = np.array([[[2.0, 3.0],
                            [2.0, 3.0]]])
```

O teste verifica que a implementação da classe `MecanismoAtencao` produz exatamente esses valores:

```python
assert np.allclose(pesos_atencao, pesos_esperados)
assert np.allclose(saida, saida_esperada)
```

Se asserções passarem, a mensagem é exibida:

```text
Teste de atenção com exemplo numérico simples passou!!!
```

Esse exemplo demonstra, de forma transparente, que o código implementa corretamente a equação de atenção:

`ATTENTION(Q, K, V) = softmax((Q K^T) / sqrt(d_k)) * V`
