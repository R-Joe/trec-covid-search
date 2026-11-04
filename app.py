import re
import os
import joblib
import gdown
import streamlit as st
import certifi
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from opensearchpy import OpenSearch


# --------------------------------------------------
# Configuração da página
# --------------------------------------------------

st.set_page_config(
    page_title="TREC-COVID Search",
    page_icon="🔎",
    layout="wide"
)

st.title("TREC-COVID Search")

st.write(
    "Mecanismo de recuperação de informação para a coleção "
    "TREC-COVID — Round 1."
)


# --------------------------------------------------
# Conexão com OpenSearch
# --------------------------------------------------

@st.cache_resource
def conectar_opensearch():

    config = st.secrets["opensearch"]

    client = OpenSearch(
        hosts=[{
            "host": config["host"],
            "port": int(config["port"])
        }],
        http_auth=(
            config["user"],
            config["password"]
        ),
        use_ssl=True,
        verify_certs=True,
        ca_certs=certifi.where()
    )

    return client


# --------------------------------------------------
# Download e carregamento dos arquivos TF-IDF
# --------------------------------------------------

ARQUIVOS = {
    "matrix": (
        "https://drive.google.com/uc?id="
        "1xXf5muTVFW6pGY9pIHYcx0pm6s0app2Q"
    ),
    "vectorizer": (
        "https://drive.google.com/uc?id="
        "13jAhFG0bZvycyfWm4qGr6rb1RSdDUot-"
    ),
    "documents": (
        "https://drive.google.com/uc?id="
        "1hYWEhcbK-qo1cWYtiqP32SOG2h2JpXMR"
    )
}


@st.cache_resource(show_spinner="Carregando modelo TF-IDF...")
def carregar_tfidf():

    os.makedirs("tfidf_artifacts", exist_ok=True)

    caminhos = {
        "matrix": "tfidf_artifacts/tfidf_matrix.npz",
        "vectorizer": "tfidf_artifacts/tfidf_vectorizer.joblib",
        "documents": "tfidf_artifacts/documents.parquet"
    }

    for chave, url in ARQUIVOS.items():

        caminho = caminhos[chave]

        if not os.path.exists(caminho):

            gdown.download(
                url,
                caminho,
                quiet=False
            )

        if not os.path.exists(caminho) or os.path.getsize(caminho) == 0:
            raise RuntimeError(
                f"Falha no download do arquivo: {chave}"
            )

    matrix = load_npz(caminhos["matrix"])
    vectorizer = joblib.load(caminhos["vectorizer"])
    documents = pd.read_parquet(caminhos["documents"])

    if matrix.shape[0] != len(documents):
        raise ValueError("Matriz e documentos possuem dimensões incompatíveis.")

    if matrix.shape[1] != len(vectorizer.vocabulary_):
        raise ValueError("Matriz e vetorizador possuem dimensões incompatíveis.")

    return matrix, vectorizer, documents


# --------------------------------------------------
# Pré-processamento da consulta TF-IDF
# --------------------------------------------------

stop_words = set(ENGLISH_STOP_WORDS)


def tokenize(text):

    tokens = re.findall(r"[a-z0-9]+", str(text).lower())

    return [
        token for token in tokens
        if token not in stop_words
    ]


# --------------------------------------------------
# Pesquisa BM25
# --------------------------------------------------

def pesquisar_bm25(client, consulta, quantidade=10):
    resposta = client.search(
        index="trec-covid-r1-bm25-v1",
        body={
            "size": quantidade,
            "track_total_hits": True,
            "_source": ["cord_uid", "title", "abstract"],
            "query": {
                "match": {
                    "text": {
                        "query": consulta
                    }
                }
            }
        },
        request_timeout=60
    )

    resultados = []

    for hit in resposta["hits"]["hits"]:
        source = hit["_source"]

        resultados.append({
            "cord_uid": source["cord_uid"],
            "title": source["title"],
            "abstract": source["abstract"],
            "score": hit["_score"]
        })

    total = resposta["hits"]["total"]["value"]

    return resultados, total

# --------------------------------------------------
# Pesquisa TF-IDF vetorial
# --------------------------------------------------

def pesquisar_tfidf(
    consulta,
    matrix,
    vectorizer,
    documents,
    quantidade=10
):

    consulta_processada = " ".join(tokenize(consulta))

    if not consulta_processada:
        return pd.DataFrame(), 0

    vetor_consulta = vectorizer.transform([consulta_processada])

    if vetor_consulta.nnz == 0:
        return pd.DataFrame(), 0

    scores = cosine_similarity(
        vetor_consulta,
        matrix
    ).ravel()

    indices = np.argsort(-scores, kind="stable")[:quantidade]

    resultados = documents.iloc[indices].copy()

    resultados["score"] = scores[indices]

    resultados = resultados.reset_index(drop=True)

    total = int(np.count_nonzero(scores > 0))

    return resultados, total


# --------------------------------------------------
# Interface
# --------------------------------------------------

consulta = st.text_input(
    "Digite sua consulta em inglês:",
    placeholder="Ex.: coronavirus origin"
)

modelo = st.radio(
    "Modelo de recuperação:",
    ["BM25", "TF-IDF"],
    horizontal=True
)

quantidade = st.selectbox(
    "Quantidade de resultados:",
    [5, 10, 20],
    index=1
)


if st.button("Pesquisar", type="primary"):

    if not consulta.strip():

        st.warning("Digite uma consulta para pesquisar.")

    else:

        try:

            with st.spinner("Pesquisando documentos..."):

                if modelo == "BM25":

                    client = conectar_opensearch()

                    resultados, total = pesquisar_bm25(
                        client,
                        consulta,
                        quantidade
                    )

                else:

                    matrix, vectorizer, documents = carregar_tfidf()

                    resultados, total = pesquisar_tfidf(
                        consulta,
                        matrix,
                        vectorizer,
                        documents,
                        quantidade
                    )

            st.subheader("Resultados da busca")

            st.write(f"Modelo: **{modelo}**")
            st.write(f"Documentos encontrados: **{total:,}**")

            if len(resultados) == 0:

                st.info("Nenhum documento encontrado.")

            else:

                for posicao, resultado in enumerate(
                    resultados.to_dict("records")
                    if isinstance(resultados, pd.DataFrame)
                    else resultados,
                    start=1
                ):

                    if modelo == "BM25":

                        doc = resultado
                        score = resultado["score"]

                    else:

                        doc = resultado
                        score = resultado["score"]

                    titulo = doc.get("title") or "Título não disponível"
                    resumo = doc.get("abstract") or "Resumo não disponível"

                    with st.container(border=True):

                        st.markdown(
                            f"**{posicao}. {titulo}**"
                        )

                        st.caption(
                            f"cord_uid: {doc.get('cord_uid', '')} "
                            f"| Score: {score:.4f}"
                        )

                        st.write(resumo[:1200])

        except Exception as erro:

            st.error(
                "Não foi possível realizar a pesquisa. "
                "Verifique a conexão com os serviços e os arquivos."
            )

            st.exception(erro)


st.divider()

st.caption(
    "Projeto acadêmico — Recuperação de Informação | "
    "TREC-COVID Round 1"
)
