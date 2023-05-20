import os
from typing import Any, Dict, List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
import pinecone


pinecone.init(
    api_key=os.environ["bd71ed61-88b4-4743-9562-38913fb1a81c"],
    environment=os.environ["us-west1-gcp-free"],
)

INDEX_NAME = "langchain-doc-index"


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["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"])
    docsearch = Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )
    return qa({"question": query, "chat_history": chat_history})
