import json
import os
import getpass
import re
from datetime import datetime
from typing import List, Dict, Any

# Импорты для работы с LangChain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Импорты для работы с окружением и безопасностью
from dotenv import load_dotenv
from getpass import getpass

# Настройка логгирования
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from dataclasses import dataclass

def format_order_details(info: dict) -> dict:
    #Форматирование данных заказа для вывода покупателю
    formatted = {}
    status = info.get("status")
    if status == "in_transit":
        eta = info.get("eta_days", 0)
        carrier = info.get("carrier", "неизвестен")
        detail = f"Заказ в пути. Ожидаемая доставка через {eta} дн. Перевозчик: {carrier}."
    elif status == "delivered":
        delivered_at = info.get("delivered_at", "не указана")
        try:
            # Опционально: можно отформатировать дату красиво
            date_obj = datetime.strptime(delivered_at, "%Y-%m-%d")
            delivered_at = date_obj.strftime("%d.%m.%Y")
        except ValueError:
            pass  # оставляем как есть, если не удалось распарсить
        detail = f"Заказ доставлен {delivered_at}."
    elif status == "processing":
        note = info.get("note", "Без примечаний")
        detail = f"Заказ в обработке. {note}"
    else:
        detail = f"Статус заказа: {status}." if status else "Информация о заказе недоступна."

    return detail

@dataclass
class ModelConfig:
    """Конфигурация для LLM и embeddings"""
    api_key: str
    base_url: str = ""
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-3.5-turbo-instruct"
    temperature: float = 0.3

def setup_api_config() -> ModelConfig:
    """Настройка API ключа с приоритетом: переменные окружения -> .env файл -> ручной ввод"""
    # Загружаем .env файл, если он существует
    load_dotenv()
    
    # Проверяем переменные окружения
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_API_BASE_URL", "")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo-instruct")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        
    return ModelConfig(
        api_key=api_key,
        base_url=base_url,
        embedding_model=embedding_model,
        llm_model=llm_model,
        temperature=temperature
    )

def create_vector_store(texts: List[str], config: ModelConfig) -> FAISS:
    """Создает векторное хранилище с настраиваемой моделью эмбеддингов"""
    try:
        # Настройка embedding модели
        embedding_kwargs = {
            "api_key": config.api_key,
            "model": config.embedding_model
        }
        
        if config.base_url:
            embedding_kwargs["base_url"] = config.base_url
        
        embeddings = OpenAIEmbeddings(**embedding_kwargs)
        
        # Создаем документы из текстов
        documents = [Document(page_content=text) for text in texts]
        
        # Разбиваем текст на чанки (для реальных приложений)
        text_splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separator="\n"
        )
        docs = text_splitter.split_documents(documents)
        
        # Создаем векторное хранилище
        vector_store = FAISS.from_documents(docs, embeddings)
        logger.info(f"Создано векторное хранилище с {len(docs)} документами")
        return vector_store
    
    except Exception as e:
        logger.error(f"Ошибка при создании векторного хранилища: {str(e)}")
        raise



def main():
    """Основная функция приложения"""
    try:
        # Настройка API ключа
        config = setup_api_config()
        logger.info(config)
        
        llm_kwargs = {
            "api_key": config.api_key,
            "temperature": config.temperature,
            "model_name": config.llm_model
        }

        if config.base_url:
            llm_kwargs["openai_api_base"] = config.base_url

        # Данные для индексации
        with open('data/faq.json', mode='r', encoding='utf-8') as file:
            faq = json.load(file)
        text_data = [f"Вопрос:'{qa['q']}'\nОтвет:'{qa['a']}'" for qa in faq]

        #Данные о заявках
        with open('data/orders.json', mode='r', encoding="utf-8") as file:
            orders = json.load(file)
        
        # Создаем векторное хранилище
        vector_store = create_vector_store(text_data, config)
        
        # Настройка retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}  # Возвращаем 2 наиболее релевантных результата
        )
        llm = ChatOpenAI(**llm_kwargs)

        # Кастомный промпт для более качественных ответов
        qa_prompt = PromptTemplate(
            template="""
            Ты — консультант магазина Shoply, отвечай кратко и вежливо. 
            Используй только предоставленный контекст для ответа.
            Если информации нет — скажи, что не знаешь.
            
            Контекст: {context}
            
            Вопрос: {input}
            Ответ:""",
            input_variables=["context", "input"]
        )

        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        def invoke_qa(query: str):
            response = retrieval_chain.invoke({"input": query})
            return {
                "result": response["answer"],
                "source_documents": response.get("context", [])
            }
        
        # Получаем запрос от пользователя
        print("\n" + "="*50)
        print("Введите 'exit' для выхода")
        print("="*50)
        
        while True:
            query = input("\nВаш вопрос: ").strip()
            if query.lower() in ['exit', 'quit', 'выйти']:
                print("До свидания! 🐱")
                break
                
            if not query:
                continue

            if query.startswith("/order"):
                match = re.fullmatch(r'/order\s+(\d+)', query.strip())
                if order := orders.get(match.group(1)):
                    print(f'Заказ #{match.group(1)}: {format_order_details(order)}')
                else:
                    print('Пожалуйста, проверьте введенные данные.')
                continue
                
            try:
                # Получаем ответ
                response = invoke_qa(query)
                print("\nОтвет:", response["result"])
                    
            except Exception as e:
                logger.error(f"Ошибка при обработке запроса: {str(e)}")
                print("Произошла ошибка при обработке запроса. Попробуйте еще раз.")
                
    except Exception as e:
        logger.critical(f"Критическая ошибка: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()