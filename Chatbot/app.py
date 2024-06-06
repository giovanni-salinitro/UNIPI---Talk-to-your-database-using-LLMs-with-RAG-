import ClickChain
import PostgreSQLChain
import streamlit as st

st.title("✨💬 Evision Chatbot")
st.caption("🚀 Fai una domanda sui tuoi dati!")

with st.sidebar:
    st.title('📜💬 Evision Chatbot')
    st.caption("📍Benvenuto nel Chatbot per parlare con i tuoi dati")

    st.subheader('Parametri')
    selected_db = st.sidebar.selectbox('Scegli il Database a cui connetterti', ['ClickHouse:Dati venduto','PostgreSQL:Anagrafica Prodotti'], key='selected_db')
    selected_selector = st.sidebar.selectbox('Scegli il selettore di esempi', ['CUSTOM SELECTOR','NGRAM','SEMANTIC SIMILARITY'], key='selected_selector')
    if selected_selector == 'NGRAM':
        st.caption("Valore per il selettore NGRAM : più è alto, meno esempi verranno scelti.")
        threshold_value = st.sidebar.slider("Valore Threshold", min_value=0.00, max_value=0.1, value=0.07, step=0.01)
    elif selected_selector == 'SEMANTIC SIMILARITY':
        st.caption("Valore per il selettore SEMANTIC SIMILARITY : il valore corrisponde al numero di esempi scelti.")
        k_value = st.sidebar.slider("Valore K", min_value=0, max_value=10, value=5, step=1)
    elif selected_selector == 'CUSTOM SELECTOR':
        st.caption("Selettore personalizzato per il database Clickhouse (Per PostgreSQL usare un altro selettore)")
    
    st.markdown('📖 Impara come funziona questo chatbot in questo [sito](https://python.langchain.com/docs/use_cases/sql/quickstart)!')

#SET UP PARAMETERS
if selected_db == 'ClickHouse:Dati venduto' :
    if selected_selector == 'NGRAM':
        ClickChain.set_threshold(threshold_value)
        ClickChain.set_NGRAM_prompt()
    elif selected_selector == 'SEMANTIC SIMILARITY':
        ClickChain.set_k(k_value)
        ClickChain.set_SEMSIM_prompt()
    elif selected_selector == 'CUSTOM SELECTOR':
        ClickChain.set_CUSTOMSELECTOR_prompt()
elif selected_db == 'PostgreSQL:Anagrafica Prodotti' :
    if selected_selector in ['NGRAM']:
        PostgreSQLChain.set_threshold(threshold_value)
        PostgreSQLChain.set_NGRAM_prompt()
    elif selected_selector == 'SEMANTIC SIMILARITY':
        PostgreSQLChain.set_k(k_value)
        PostgreSQLChain.set_SEMSIM_prompt()
    elif selected_selector == 'CUSTOM SELECTOR':
        PostgreSQLChain.set_threshold(0.01)
        PostgreSQLChain.set_NGRAM_prompt()
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Come posso aiutarti oggi?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Come posso aiutarti oggi?"}]
st.sidebar.button('Cancella cronologia chat', on_click=clear_chat_history)

# User-provided prompt
if prompt := st.chat_input("Digita qui la tua richiesta.."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if selected_db == 'ClickHouse:Dati venduto':
                response = ClickChain.get_response(prompt)
            elif selected_db == 'PostgreSQL:Anagrafica Prodotti':
                response = PostgreSQLChain.get_response(prompt)
            #msg = response['result'].replace(':\n                     ','-')
            msg1 = response['result'].replace(':',' : ')
            placeholder = st.empty()
            full_response = ''
            for item in msg1:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)


