# Interface de Comunicação com IA (Chat) - Usando Streamlit e LLMs (langchain e hunggingface) + RAG


<b>Autor: Bruno P. Iglesias</b>

## Descrição

Este repositório contém uma aplicação desenvolvida com **Streamlit** que permite aos usuários fazer o upload de arquivos PDF e interagir com **Modelos de Linguagem (LLMs)**. A aplicação utiliza técnicas de **retrieval-augmented generation (RAG)**, que combinam recuperação de informações e geração de respostas, para fornecer interações contextuais baseadas no conteúdo dos documentos enviados.

<img src="https://github.com/bruiglesias/llm-langchain-and-hunggingface/blob/master/img_01.png" />

## Funcionalidades

- **Upload de arquivos PDF**: Os usuários podem enviar vários arquivos PDF para análise.
- **Divisão de documentos**: Os PDFs são automaticamente divididos em pedaços menores de texto para processamento mais eficiente.
- **Geração de respostas**: O modelo responde às consultas com base no conteúdo dos documentos fornecidos.
- **Suporte a diferentes LLMs**: A aplicação oferece suporte para diferentes modelos de IA, como **Hugging Face**, **OpenAI** e **Ollama**.
- **Histórico de conversas**: O histórico de mensagens entre o usuário e o modelo é mantido para contextualizar as respostas.

## Pré-requisitos

- **Python 3.8+**
- **Streamlit**
- **Hugging Face Transformers**
- **FAISS**
- **PyPDF2**
- **Tempfile**

Certifique-se de instalar todas as dependências necessárias executando:

```bash
pip install -r requirements.txt
```

## Como Executar a Aplicação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
```
2. Instale as dependências:
```bash
pip install -r requirements.txt
```
3. Execute o Streamlit:
```bash
streamlit run projeto-3-conversa-com-documentos.py
```
4. Acesse a aplicação no seu navegador na URL gerada, geralmente http://localhost:8501.

## Estrutura do Código
1. Configuração do Streamlit:
```python
st.set_page_config(page_title="Seu modelo de linguagem", page_icon="")
st.title("Interface de Comunicação com IA")
```
Este trecho configura o título da página e o cabeçalho principal da aplicação.
2. Upload de Arquivos PDF:
```python
uploads = st.sidebar.file_uploader(
    label="Enviar arquivos", type=['pdf'],
    accept_multiple_files=True
)

if not uploads:
    st.info("Por favor, envie algum arquivo para continuar")
    st.stop()

```
Os usuários podem enviar múltiplos arquivos PDF pela barra lateral da interface.
3. Processamento dos Documentos e Indexação:
```python
def config_retriever(uploads):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploads:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, 'wb') as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())
    
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_spliter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local('vectorstore/db_faiss')

    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 3, 'fetch_k': 4})

    return retriever

```
- PyPDFLoader carrega e processa os PDFs.
- FAISS cria um índice para busca eficiente por similaridade usando embeddings gerados com modelos Hugging Face.

4. Geração de Respostas com Diferentes Modelos:
Dependendo da escolha do usuário, a aplicação pode utilizar modelos da Hugging Face, OpenAI ou Ollama:
```python
def model_hf_hub(model = "meta-llama/Meta-Llama-3-8B-Instruct", temperature = 0.1):
    llm = HuggingFaceHub(
        repo_id = model,
        model_kwargs={
            "temperature": temperature,
            "return_full_text": False,
            "max_new_tokens": 512,
        })
    return llm

def model_openai(model = "gpt-4o-mini", temperature = 0.1):
    llm = ChatOpenAI(model = model, temperature = temperature)
    return llm

def model_ollama(model = "phi3", temperature = 0.1):
    llm = ChatOllama(model = model, temperature = temperature)
    return llm

```
5. Configuração da Cadeia de Recuperação e Geração (RAG):
```python
    def config_rag_chain(model_class, retriever):
    if model_class == "hf_hub":
        llm = model_hf_hub()
    elif model_class == "openai":
        llm = model_openai()
    elif model_class == "ollama":
        llm = model_ollama()

    qa_prompt_template = """Você é um assistente virtual prestativo e está respondendo perguntas gerais.
    Use os seguintes pedaços de contexto recuperado para responder à pergunta.
    Responda em português. \n\n
    Pergunta: {input} \n
    Contexto: {context}"""

    qa_prompt = PromptTemplate.from_template(qa_prompt_template)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    return rag_chain

```
O código acima configura a cadeia RAG que combina a busca por contexto e a geração de respostas.
6. Exibição de Mensagens na Interface:
```python
user_query = st.chat_input("Digite sua mensagem aqui...")

if user_query is not None and user_query != "" and uploads is not None:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        if st.session_state.docs_list != uploads:
            st.session_state.docs_list = uploads
            st.session_state.retriever = config_retriever(uploads)

        rag_chain = config_rag_chain(model_class, st.session_state.retriever)
        result = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})

        resp = result['answer']
        st.write(resp)

```
Este código exibe a mensagem do usuário e a resposta da IA na interface do Streamlit.

## Personalização

- Modelos: Você pode ajustar o modelo de linguagem mudando os parâmetros dos métodos model_hf_hub, model_openai, ou model_ollama.
- Prompt: O prompt de geração de respostas pode ser adaptado conforme necessário dentro da função config_rag_chain.


Sinta-se à vontade para adaptar e melhorar este README para melhor refletir suas necessidades específicas ou alterações no projeto!
