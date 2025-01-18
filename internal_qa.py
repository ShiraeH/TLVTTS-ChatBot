import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import StreamlitCallbackHandler
from add_document import initialize_vectorstore
from langchain.chains import RetrievalQA

load_dotenv()

logo_path = os.path.join(os.path.dirname(__file__), "LOGO.png")
st.image(logo_path)
#st.title("CMTO Chat Bot")

def create_qa_chain():
    """
    PineconeとOpenAIモデルを使用してQAチェーンを作成する
    """
    vectorstore = initialize_vectorstore()
    callback = StreamlitCallbackHandler(st.container())

    llm = ChatOpenAI(
        model_name=os.environ["OPENAI_API_MODEL"],
        temperature=float(os.environ["OPENAI_API_TEMPERATURE"]),
        stored = False,
        streaming=True,
        callbacks=[callback],
    )

    qa_chain = RetrievalQA.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    return qa_chain

# 初回セッション時にメッセージ履歴を初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# 過去のメッセージを表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 新しいプロンプトを入力
prompt = st.chat_input("please enter your question...")

if prompt:
    # ユーザーの入力を履歴に追加
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # 履歴全体を結合してコンテキストとして生成
    conversation_history = "\n".join(
        [f"{message['role']}: {message['content']}" for message in st.session_state.messages]
    )

    # アシスタントの応答を生成
    with st.chat_message("assistant"):
        qa_chain = create_qa_chain()
        response = qa_chain.invoke(conversation_history)

    # アシスタントの応答を履歴に追加
    st.session_state.messages.append({"role": "assistant", "content": response["result"]})

    # アシスタントの応答を表示
    st.chat_message("assistant").markdown(response["result"])
