# llm.py

import os
import logging
from typing import Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate,
)
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# (프로젝트에 존재한다고 하셨던) 예시 세트
from config import answer_examples

store: Dict[str, BaseChatMessageHistory] = {}
logging.basicConfig(level=logging.INFO)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or create a chat message history for a given session ID."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_retriever() -> PineconeVectorStore:
    """Create a retriever using Ollama embeddings and Pinecone vector store."""
    try:
        # ⚠️ embedmodel 예: "nomic-embed-text" (768차원)
        # Pinecone 인덱스의 dimension도 768이어야 합니다.
        embedding = OllamaEmbeddings(
            model=os.getenv("embedmodel") or "nomic-embed-text"
        )
        index_name = "tax-markdown-index"

        database = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embedding,
        )
        retriever = database.as_retriever(search_kwargs={"k": 4})
        return retriever
    except Exception as e:
        logging.error(f"Error creating retriever: {e}")
        raise


def get_llm(model: str | None = None) -> ChatOllama:
    """Instantiate Ollama chat model (로컬 ollama가 실행 중이어야 합니다)."""
    model = model or os.getenv("model") or "llama3.2:latest"
    return ChatOllama(model=model)


def get_history_retriever():
    """Create a history-aware retriever that can contextualize questions based on chat history."""
    llm = get_llm()
    retriever = get_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever


def get_dictionary_chain() -> StrOutputParser:
    """Create a chain that uses a dictionary to modify user questions."""
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        """
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전:
        {dictionary}
        
        질문: {question}
"""
    )

    dictionary_chain = prompt | llm | StrOutputParser()
    return dictionary_chain


def get_rag_chain() -> RunnableWithMessageHistory:
    """Create a retrieval-augmented generation (RAG) chain for answering questions."""
    llm = get_llm()

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )

    system_prompt = (
        "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요"
        "아래에 제공된 문서를 활용해서 답변해주시고"
        "답변을 알 수 없다면 모른다고 답변해주세요"
        "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주시고"
        "2-3 문장정도의 짧은 내용의 답변을 원합니다"
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = get_history_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick("answer")

    return conversational_rag_chain


def get_ai_response(user_message: str) -> str:
    """Generate an AI response to a user message using dictionary and RAG chains."""
    try:
        dictionary_chain = get_dictionary_chain()

        rag_chain = get_rag_chain()

        tax_chain = {"input": dictionary_chain} | rag_chain

        fragments: list[str] = []
        for chunk in tax_chain.stream(
            {
                "question": user_message,
                "dictionary": "\n".join(["사람을 나타내는 표현 -> 거주자"]),
            },
            config={"configurable": {"session_id": "abc123"}},
        ):
            if isinstance(chunk, str):
                fragments.append(chunk)
            elif (
                isinstance(chunk, dict)
                and "answer" in chunk
                and isinstance(chunk["answer"], str)
            ):
                fragments.append(chunk["answer"])

        if fragments:
            return "".join(fragments).strip()

        result = tax_chain.invoke(
            {
                "question": user_message,
                "dictionary": "\n".join(["사람을 나타내는 표현 -> 거주자"]),
            },
            config={"configurable": {"session_id": "abc123"}},
        )
        if isinstance(result, str):
            return result
        if isinstance(result, dict) and "answer" in result:
            return str(result["answer"])
        return str(result)

    except Exception as e:
        logging.error(f"Error generating AI response: {e}")
        return "An error occurred while processing your request."
