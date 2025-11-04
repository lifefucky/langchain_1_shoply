import argparse
import json

import logging
import os
import re
import sys
import traceback
from datetime import datetime
from typing import List, Dict, Any, Union

from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Конфигурация для LLM и embeddings"""
    api_key: str
    base_url: str = ""
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.3
    context_length: int = 3
    brand_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразует конфигурацию в словарь с маскировкой API ключа.
        Возвращает:
            Словарь с безопасными данными конфигурации
        """
        return {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "context_length": self.context_length
        }


def setup_api_config() -> ModelConfig:
    """Настройка API ключа с приоритетом: переменные окружения -> .env файл -> ручной ввод"""
    # Загружаем .env файл, если он существует
    load_dotenv()

    # Проверяем переменные окружения
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_API_BASE_URL")
    embedding_model = os.getenv("EMBEDDING_MODEL")
    llm_model = os.getenv("LLM_MODEL")
    temperature = float(os.getenv("LLM_TEMPERATURE"))
    context_length = int(os.getenv("CONTEXT_LENGTH"))
    brand_name = os.getenv("BRAND_NAME")

    return ModelConfig(
        api_key=api_key,
        base_url=base_url,
        embedding_model=embedding_model,
        llm_model=llm_model,
        temperature=temperature,
        context_length=context_length,
        brand_name=brand_name
    )


class Consultant:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data")
        with open(os.path.join(data_dir, "faq.json"), mode='r', encoding='utf-8') as file:
            self.faq = json.load(file)
        with open(os.path.join(data_dir, "orders.json"), mode='r', encoding="utf-8") as file:
            self.orders = json.load(file)

        os.makedirs("logs", exist_ok=True)
        now: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = self.setup_logger(f"logs/session_{now}.jsonl")

        self.conversation_history: list[dict[str, str]] = []
        self.context_length = self.model_config.context_length

    @staticmethod
    def setup_logger(log_file):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

        return logger

    def format_conversation_history(self) -> str:
        """Форматирует историю диалога в строку"""
        return "\n".join(
            [f"{msg['role'].upper()}: {msg['content']}"
             for msg in self.conversation_history[-self.context_length:]]
        )

    def add_to_history(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})

    def prepare_text_faq(self) -> List[str]:
        return [f"Вопрос:'{qa['q']}'\nОтвет:'{qa['a']}'" for qa in self.faq]

    def create_vector_store(self) -> FAISS:
        texts = self.prepare_text_faq()

        try:
            # Настройка embedding модели
            embedding_kwargs = {
                "api_key": self.model_config.api_key,
                "model": self.model_config.embedding_model
            }

            if self.model_config.base_url:
                embedding_kwargs["base_url"] = self.model_config.base_url

            embeddings = OpenAIEmbeddings(**embedding_kwargs)

            documents = [Document(page_content=text) for text in texts]

            text_splitter = CharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separator="\n"
            )
            docs = text_splitter.split_documents(documents)

            vector_store = FAISS.from_documents(docs, embeddings)
            self.add_log(event='vector_store_created', document_count=len(docs), chunks_count=len(docs))
            return vector_store

        except Exception as e:
            details = {
                    "documents_count": len(texts) if 'texts' in locals() else 0
                }
            self.add_log(type='error', message=str(e), details=details, event_type='vector_store_creation')
            raise

    def retrieval_chain(self, model: ChatOpenAI, vector_store: FAISS):
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        qa_prompt = PromptTemplate(
            template="""
                    Ты — консультант магазина {brand_name}, отвечай кратко и вежливо. 
                    Используй только предоставленный контекст для ответа.
                    Если информации нет — скажи, что не знаешь.

                    Контекст: {context}
                    
                    История диалога: 
                    {history}

                    Вопрос: {input}
                    Ответ:""",
            input_variables=["brand_name", "context", "input", "history"]
        )
        document_chain = create_stuff_documents_chain(model, qa_prompt)
        return create_retrieval_chain(retriever, document_chain)

    def faq_processor(self, query: str, retrieval_chain):
        try:
            with get_openai_callback() as cb:
                response = retrieval_chain.invoke({
                    "brand_name": self.model_config.brand_name,
                    "input": query,
                    "history": self.format_conversation_history()
                })
                serializable_response = {
                    "answer": response.get("answer", ""),
                    "usage": {"total_tokens": cb.total_tokens, "prompt_tokens": cb.prompt_tokens, "completion_tokens": cb.completion_tokens},
                    "context": ""
                }

            self.add_log(query=query, message=serializable_response)
            return response["answer"]
        except Exception as e:
            self.add_log(type='error', query=query, message=str(e), event='faq_error', event_type='faq_processing')
            print("Произошла ошибка при обработке запроса. Попробуйте еще раз.")

    def orders_processor(self, query: str):
        match = re.fullmatch(r'/order\s+(\d+)', query.strip())
        if not match:
            response = 'Неверный формат. Используйте: /order <номер>'
            self.add_log(type='error', query=query, message=response, event='order_error', event_type='invalid_format')
            return response

        order_id = match.group(1)
        if order := self.orders.get(order_id):
            response = f'Заказ #{match.group(1)}: {format_order_details(order)}'
            self.add_log(query=query, message=response)
            return response
        else:
            response = 'Пожалуйста, проверьте введенные данные.'
            self.add_log(type='error', query=query, message=response, event='order_error', event_type='not_found')
            return response

    def add_log(self, type: str = "info", query: str = None, message: Union[str, dict] = None, **kwargs):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "message": message,
            **kwargs
        }
        if type == 'error':
            self.logger.error(json.dumps(log_entry, ensure_ascii=False))
        else:
            self.logger.info(json.dumps(log_entry, ensure_ascii=False))


def format_order_details(info: dict) -> str:
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


def main():
    parser = argparse.ArgumentParser(description="Consultant Bot")
    parser.add_argument('--url', type=str, help='Base URL for LLM API')
    parser.add_argument('--model', type=str, help='LLM model name')
    parser.add_argument('--api-key', type=str, help='API key for authentication')
    args = parser.parse_args()

    try:
        model_config = setup_api_config()
        if args.api_key:
            model_config.api_key = args.api_key
        if args.url:
            model_config.base_url = args.url
        if args.model:
            model_config.llm_model = args.model

        bot = Consultant(model_config=model_config)
        config = bot.model_config
        bot.add_log(event="config_loaded", config=config.to_dict())

        llm_kwargs = {
            "api_key": config.api_key,
            "temperature": config.temperature,
            "model_name": config.llm_model,
            "openai_api_base": config.base_url}
        model = ChatOpenAI(**llm_kwargs)

        # Создаем векторное хранилище
        vector_store = bot.create_vector_store()
        retrieval_chain = bot.retrieval_chain(model=model, vector_store=vector_store)

        # Получаем запрос от пользователя
        print("\n" + "=" * 50)
        print("Введите 'exit' для выхода")

        while True:
            query = input("\nВаш вопрос: ").strip()
            bot.add_to_history("user", query)
            if query.lower() in ['exit', 'quit', 'выйти']:
                print("До свидания! 🐱")
                bot.add_log(message="Пользователь инициировал выход.")
                break

            if not query:
                continue

            if query.startswith("/order"):
                response = bot.orders_processor(query=query)
                print(response)
                bot.add_to_history("assistant", response)
                continue

            response = bot.faq_processor(query=query, retrieval_chain=retrieval_chain)
            print(response)
            bot.add_to_history("assistant", response)

    except Exception as e:
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "critical_error",
            "error": str(e),
            "traceback": traceback.format_exc() if 'traceback' in sys.modules else None
        }
        logging.getLogger(__name__).error(json.dumps(error_entry, ensure_ascii=False))
        print("Критическая ошибка приложения. Детали записаны в лог.")
        exit(1)


if __name__ == "__main__":
    main()
