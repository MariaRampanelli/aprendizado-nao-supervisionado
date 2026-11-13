
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# =====================================
# 1. CARREGAR OS DADOS
# =====================================

dados = pd.read_csv("MusicasJogosDataset_20232_20262.csv")

# Separar nomes e preferências
nomes = dados.iloc[:, 0]
preferencias = dados.iloc[:, 1:]

print("Quantidade de pessoas:", len(dados))
print("Quantidade de preferências:", preferencias.shape[1])


# =====================================
# 2. TESTAR K-MEANS
# =====================================

print("\nRESULTADOS DO K-MEANS")
print("------------------------------------")

for k in range(2, 11):

    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    grupos = modelo.fit_predict(preferencias)

    silhueta = silhouette_score(
        preferencias,
        grupos
    )

    print(f"{k} grupos | Silhueta: {silhueta:.4f}")


# =====================================
# 3. TESTAR AGRUPAMENTO HIERÁRQUICO
# =====================================

print("\nRESULTADOS DO AGRUPAMENTO HIERÁRQUICO")
print("------------------------------------")

for metodo in ["ward", "complete", "average", "single"]:

    print(f"\nMétodo: {metodo}")

    for k in range(2, 11):

        modelo = AgglomerativeClustering(
            n_clusters=k,
            linkage=metodo
        )

        grupos = modelo.fit_predict(preferencias)

        silhueta = silhouette_score(
            preferencias,
            grupos
        )

        print(f"{k} grupos | Silhueta: {silhueta:.4f}")

# =====================================
# 4. ANALISAR OS GRUPOS
# =====================================

print("\nANÁLISE DOS GRUPOS")
print("------------------------------------")

# Configurações selecionadas
kmeans_final = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

hierarquico_final = AgglomerativeClustering(
    n_clusters=2,
    linkage="average"
)

# Identificar o grupo de cada pessoa
grupos_kmeans = kmeans_final.fit_predict(preferencias)
grupos_hierarquico = hierarquico_final.fit_predict(preferencias)

# Criar tabela com resultados
resultado = pd.DataFrame({
    "Nome": nomes.astype(str).str.strip(),
    "Grupo_KMeans": grupos_kmeans,
    "Grupo_Hierarquico": grupos_hierarquico
})

# Salvar a classificação de cada pessoa
resultado.to_csv(
    "resultado_grupos.csv",
    index=False
)

# Separar colunas de músicas e jogos
musicas = list(preferencias.columns[:21])
jogos = list(preferencias.columns[21:])

# Analisar cada algoritmo
for nome_algoritmo, grupos in [
    ("K-Means", grupos_kmeans),
    ("Hierarquico", grupos_hierarquico)
]:
    print(f"\nALGORITMO: {nome_algoritmo}")

    tabela = preferencias.copy()
    tabela["Grupo"] = grupos

    # Quantidade de pessoas em cada grupo
    print("\nQuantidade de pessoas por grupo:")
    print(tabela["Grupo"].value_counts().sort_index())

    # Centroide: média de cada preferência por grupo
    centroides = tabela.groupby("Grupo").mean()

    # Salvar os centroides
    centroides.to_csv(
        f"centroides_{nome_algoritmo}.csv"
    )

    for grupo in centroides.index:
        centroide = centroides.loc[grupo]

        print(f"\nGRUPO {grupo}")

        print("\nMúsicas mais populares:")
        print(
            centroide[musicas]
            .sort_values(ascending=False)
            .head(5)
        )

        print("\nJogos mais populares:")
        print(
            centroide[jogos]
            .sort_values(ascending=False)
            .head(5)
        )


# =====================================
# 6. LOCALIZAR INTEGRANTES
# =====================================

print("\nGRUPOS DAS INTEGRANTES")
print("------------------------------------")

# Nomes exatos cadastrados na planilha
integrantes = [
    "Maria Rampanelli",
    "Isadora Sbeghen de Campos"
]

# Remover espaços extras dos nomes
resultado["Nome"] = resultado["Nome"].str.strip()

# Buscar somente as integrantes da dupla
encontrados = resultado[
    resultado["Nome"].isin(integrantes)
]

print(encontrados.to_string(index=False))

# Verificar se alguma integrante não foi encontrada
for integrante in integrantes:
    if integrante not in encontrados["Nome"].values:
        print(f"Integrante não encontrada: {integrante}")
