import random
import csv 


X = []          # lista de entradas [x1, x2, x3]
d = []          # saídas desejadas (-1 ou 1)
nome_do_arquivo = 'base.csv' 

try:
    with open(nome_do_arquivo, mode='r', encoding='utf-8') as arquivo:
        # Se os dados não carregarem com vírgula, troque ',' por ';' na linha abaixo
        leitor = csv.reader(arquivo, delimiter=',') 
        
        next(leitor) # PULA A PRIMEIRA LINHA (O CABEÇALHO)
        
        for linha in leitor:
            # Pula linhas vazias se houver
            if not linha: 
                continue
                
            # O arquivo tem 5 colunas (índices 0 a 4)
            # 0: Instância | 1: x1 | 2: x2 | 3: x3 | 4: Classe
            if len(linha) >= 5: 
                X.append([float(linha[1]), float(linha[2]), float(linha[3])])
                d.append(int(linha[4]))
                
    print(f"Sucesso: {len(X)} instâncias carregadas do arquivo {nome_do_arquivo}.\n")

except FileNotFoundError:
    print(f"Erro: O arquivo '{nome_do_arquivo}' não foi encontrado.")
    exit()
# 2. PARÂMETROS DO TREINAMENTO
#(continue com a taxa_aprendizagem = 0.01)
taxa_aprendizagem = 0.01
max_epocas = 1000
num_treinamentos = 5
# 3. FUNÇÃO DE ATIVAÇÃO DEGRAU BIPOLAR
def ativacao(v):
    return 1 if v >= 0 else -1
# 4. LOOP PRINCIPAL DE TREINAMENTOS
resultados = []   # lista para armazenar os dados de cada treinamento
for t in range(1, num_treinamentos + 1):
    # Inicialização aleatória dos pesos e bias entre 0 e 1
    w1 = random.random()
    w2 = random.random()
    w3 = random.random()
    theta = random.random()
    # Salva cópia dos pesos iniciais
    pesos_iniciais = [w1, w2, w3] 
    epoca = 0
    houve_erro = True
    # Treinamento
    while houve_erro and epoca < max_epocas:
        houve_erro = False
        
        # Itera sobre todas as 30 amostras
        for i in range(len(X)):
            x1, x2, x3 = X[i]
            desejo = d[i]

            v = w1*x1 + w2*x2 + w3*x3 - theta
            y = ativacao(v)
            if y != desejo:
                erro = desejo - y
                # Atualização dos pesos
                w1 += taxa_aprendizagem * erro * x1
                w2 += taxa_aprendizagem * erro * x2
                w3 += taxa_aprendizagem * erro * x3
                
                # Atualização do bias (theta)
                theta -= taxa_aprendizagem * erro
                
                houve_erro = True   # ainda precisa aprender  
        epoca += 1 
    # Salva os resultados deste treinamento
    resultados.append({
        'treinamento': t,
        'pesos_iniciais': pesos_iniciais,
        'pesos_finais': [w1, w2, w3],
        'epocas': epoca
    })
# 5.TABELA FORMATADA
print("Treinamento | Pesos Iniciais (W1, W2, W3)       | Pesos Finais (W1, W2, W3)         | Épocas")
print("-" * 95)
for r in resultados:
    pi = r['pesos_iniciais']
    pf = r['pesos_finais'] 
    # Formatação com 4 casas decimais
    str_iniciais = f"{pi[0]:.4f}, {pi[1]:.4f}, {pi[2]:.4f}"
    str_finais   = f"{pf[0]:.4f}, {pf[1]:.4f}, {pf[2]:.4f}"
    print(f"{r['treinamento']:^11} | {str_iniciais:<33} | {str_finais:<32} | {r['epocas']:^6}")
