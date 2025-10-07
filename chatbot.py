from typing import List, Optional, Dict, Any
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document


class RAGChatbot:
    """RAG-based chatbot that uses retrieved documents to answer questions."""
    
    def __init__(
        self,
        retriever,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """
        Initialize the RAG chatbot.
        
        Args:
            retriever: Retriever for fetching relevant documents
            model: Name of the OpenAI model to use
            temperature: Sampling temperature for response generation
            max_tokens: Maximum tokens in the response
        """
        self.llm = ChatOpenAI(
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Custom prompt template for RAG
        template = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer: """
        
        self.qa_prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.qa_prompt}
        )
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question to the chatbot.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing the answer and source documents
        """
        result = self.chain({"question": question})
        
        return {
            "answer": result["answer"],
            "source_documents": result.get("source_documents", [])
        }
    
    def chat(self, message: str) -> str:
        """
        Simple chat interface that returns just the answer.
        
        Args:
            message: User's message
            
        Returns:
            Chatbot's response
        """
        result = self.ask(message)
        return result["answer"]
    
    def get_chat_history(self) -> List[Any]:
        """
        Get the conversation history.
        
        Returns:
            List of conversation messages
        """
        return self.memory.chat_memory.messages
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.memory.clear()
