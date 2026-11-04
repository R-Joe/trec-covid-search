

import re
import streamlit as st
import certifi

from opensearchpy import OpenSearch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


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
# Pré-processamento da consulta
# Mesmo procedimento usado na indexação
# --------------------------------------------------

stop_words = set(ENGLISH_STOP_WORDS)

def tokenize(text):
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    return [
        token for token in tokens
        if token not in stop_words
    ]


# --------------------------------------------------
# Recuperação de documentos
# --------------------------------------------------

def pesquisar(client, consulta, modelo, quantidade=10):

    consulta_processada = " ".join(tokenize(consulta))

    if not consulta_processada:
        return [], 0

    indices = {
        "BM25": "trec-covid",
        "TF-IDF": "trec-covid-tfidf"
    }

    body = {
        "size": quantidade,
        "track_total_hits": True,
        "_source": [
            "cord_uid",
            "title",
            "abstract"
        ],
        "query": {
            "match": {
                "text_processed": consulta_processada
            }
        },
        "sort": [
            {"_score": "desc"},
            {"cord_uid": "asc"}
        ]
    }

    resposta = client.search(
        index=indices[modelo],
        body=body,
        request_timeout=60
    )

    resultados = resposta["hits"]["hits"]
    total = resposta["hits"]["total"]["value"]

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
            client = conectar_opensearch()

            with st.spinner("Pesquisando documentos..."):
                resultados, total = pesquisar(
                    client,
                    consulta,
                    modelo,
                    quantidade
                )

            st.subheader("Resultados da busca")

            st.write(f"Modelo: **{modelo}**")
            st.write(f"Documentos encontrados: **{total:,}**")

            if not resultados:
                st.info("Nenhum documento encontrado.")

            for posicao, hit in enumerate(resultados, start=1):

                doc = hit["_source"]

                titulo = doc.get("title") or "Título não disponível"
                resumo = doc.get("abstract") or "Resumo não disponível"

                with st.container(border=True):

                    st.markdown(
                        f"**{posicao}. {titulo}**"
                    )

                    st.caption(
                        f"cord_uid: {doc.get('cord_uid', '')} "
                        f"| Score: {hit['_score']:.4f}"
                    )

                    st.write(resumo[:1200])

        except Exception as erro:
            st.error(
                "Não foi possível realizar a pesquisa. "
                "Verifique a conexão com o serviço."
            )
            st.exception(erro)


st.divider()

st.caption(
    "Projeto acadêmico — Recuperação de Informação | "
    "TREC-COVID Round 1"
)
