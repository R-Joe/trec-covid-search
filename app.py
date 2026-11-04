import streamlit as st
import certifi
from opensearchpy import OpenSearch

st.set_page_config(
    page_title="Diagnóstico OpenSearch",
    page_icon="🔎"
)

st.title("Diagnóstico da conexão com OpenSearch")

try:
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
        ca_certs=certifi.where(),
        timeout=20
    )

    st.success("Configuração dos Secrets carregada.")

    # Teste 1: conexão com o servidor
    if client.ping():
        st.success("Conexão com o OpenSearch estabelecida!")

        info = client.info()

        st.write("Versão do OpenSearch:")
        st.write(info["version"]["number"])

        # Teste 2: verificar os índices
        for indice in ["trec-covid", "trec-covid-tfidf"]:

            st.subheader(f"Índice: {indice}")

            if client.indices.exists(index=indice):

                resultado = client.count(index=indice)

                st.success("Índice encontrado!")

                st.write(
                    f"Quantidade de documentos: "
                    f"{resultado['count']:,}"
                )

            else:
                st.error("Índice não encontrado.")

    else:
        st.error("Não foi possível estabelecer conexão.")

except Exception as erro:
    st.error("Ocorreu um erro durante o diagnóstico.")
    st.code(str(erro))
