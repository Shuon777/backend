import os
import logging
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict
from langchain_core.tools import tool
from dotenv import load_dotenv
from infrastructure.llm_integration import get_gigachat
from langgraph.prebuilt import create_react_agent
from langchain_gigachat import GigaChat
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain.agents import AgentExecutor, Tool
from langchain.agents import initialize_agent
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

@tool
def simple_search_tool(species_name: str, search_type: str) -> str:
    """
    Ищет информацию по названию вида. 
    species_name - название вида
    Параметр search_type должен быть строго 'image' или 'text'.
    """
    if search_type not in ('image', 'text'):
        return json.dumps([{
            "error": f"Недопустимый тип поиска: {search_type}",
            "suggestions": ["Используйте 'image' для изображений или 'text' для текста"]
        }], ensure_ascii=False)
    return sql_agent_instance._execute_simple_search(species_name, search_type)

@tool
def attribute_search_tool(species_name: str, attribute: str) -> str:
    """Ищет информацию по виду с дополнительными атрибутами (цвет, место, особенность)."""
    return sql_agent_instance._execute_attribute_search(species_name, attribute)

@tool
def geo_search_tool(species_name: str, location: str, distance_km: int = 10) -> str:
    """Ищет геоданные по виду в указанной локации или рядом с ней."""
    return sql_agent_instance._execute_geo_search(species_name, location, distance_km)

@tool
def text_description_tool(species_name: str) -> str:
    """Возвращает текстовое описание вида."""
    return sql_agent_instance._execute_text_description(species_name)

class SimpleSearchArgsSchema(BaseModel):
    species_name: str
    search_type: str
    
class SQLAgent:
    def __init__(self):
        self.db_config = {
            "dbname": os.getenv("DB_NAME", "eco"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "Fdf78yh0a4b!"),
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432")
        }
        
        
        self.llm = GigaChat(
            credentials=os.getenv("GIGACHAT_CREDENTIALS"),
            verify_ssl_certs=False,
        )
        
        
        self.tools = [
            Tool(
                name="simple_search_tool",
                func=self._execute_simple_search,
                description=(
                    "Ищет информацию по названию вида. "
                    "Параметры:\n"
                    "- species_name: русское или латинское название вида\n"
                    "- search_type: 'image' для изображений или 'text' для текста\n\n"
                    "Примеры:\n"
                    "- 'Покажи даурского ежа' → simple_search_tool('даурский еж', 'image')\n"
                    "- 'Расскажи о байкальской нерпе' → simple_search_tool('байкальская нерпа', 'text')"
                ),
                args_schema=SimpleSearchArgsSchema
            )
        ]
        
        
        self.llm_with_tools = self.llm.bind_tools(
            tools=self.tools,
            tool_choice="auto"
        )
        
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Ты эксперт по Байкальской природной территории. "
                "Анализируй запросы пользователей и определяй:\n"
                "1. Название вида (species_name)\n"
                "2. Тип информации (search_type: 'image' или 'text')\n\n"
                
                "ПРАВИЛА ОПРЕДЕЛЕНИЯ ПАРАМЕТРОВ:\n"
                "1. species_name ДОЛЖНО быть:\n"
                "   - Полным названием вида в именительном падеже\n"
                "   - На русском или латинском языке\n"
                "   - Если в запросе несколько названий, выбирай основное\n"
                "   - Примеры: 'даурский ёж', 'pinus sibirica'\n\n"
                
                "2. search_type ДОЛЖЕН быть:\n"
                "   - 'image' если в запросе есть: покажи, фото, изображение, как выглядит\n"
                "   - 'text' если в запросе есть: расскажи, описание, информация, текст\n"
                "   - По умолчанию: 'image'\n\n"
                
                "КРИТИЧЕСКИ ВАЖНО:\n"
                "- Всегда извлекай ТОЧНОЕ название вида\n"
                "- Никогда не сокращай названия видов\n"
                "- Если название вида не найдено, верни пустую строку\n\n"
                
                "ПРИМЕРЫ:\n"
                "- 'Покажи фото байкальской нерпы' → species_name='байкальская нерпа', search_type='image'\n"
                "- 'Как выглядит pusa sibirica?' → species_name='pusa sibirica', search_type='image'\n"
                "- 'Расскажи о жизни ольхонской полевки' → species_name='ольхонская полевка', search_type='text'\n"
                "- 'Даурский ёж' → species_name='даурский ёж', search_type='image'\n"
                "- 'Покажи изображения' → species_name='', search_type='image'"
            )),
            ("user", "{input}"),
        ])
        
        
        self.chain = self.prompt | self.llm_with_tools

    def _execute_simple_search_wrapper(self, query: str) -> str:
        """Обертка для simple_search_tool в формате LangChain Tool"""
        
        if any(word in query.lower() for word in ["покажи", "фото", "как выглядит"]):
            search_type = "image"
        else:
            search_type = "text"
        
        
        species_name = query.split()[-1]  
        
        return self._execute_simple_search(species_name, search_type)

    def ask(self, question: str) -> str:
        try:
            logger.info(f"Обработка вопроса: {question}")
            response = self.chain.invoke({"input": question})
            
            # Добавим логирование полного ответа от LLM
            logger.debug(f"Полный ответ от LLM: {response}")
            
            # Проверяем, содержит ли ответ вызов инструмента
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_call = response.tool_calls[0]
                logger.debug(f"Вызов инструмента обнаружен: {tool_call}")
                logger.debug(f"Full tool_call object: {tool_call}")
                logger.debug(f"Response content: {response.content}")
                if tool_call['name'] == 'simple_search_tool':
                    args = tool_call['args']
                    species_name = args.get('species_name', '')
                    search_type = args.get('search_type', 'image')
                    
                    logger.info(f"Вызов инструмента: {tool_call['name']} "
                            f"с параметрами: species_name='{species_name}', "
                            f"search_type='{search_type}'")
                    
                    # Добавим проверку на пустое название вида
                    if not species_name:
                        logger.warning("Название вида не было извлечено из запроса")
                        return json.dumps({
                            "error": "Не удалось определить название вида",
                            "question": question,
                            "suggestions": [
                                "Уточните название вида",
                                "Используйте полное название на русском или латинском языке"
                            ]
                        }, ensure_ascii=False)
                    
                    result = self._execute_simple_search(species_name, search_type)
                    return json.dumps({
                        "answer": result,
                        "success": True,
                        "parameters": {
                            "species_name": species_name,
                            "search_type": search_type
                        },
                        "debug": {
                            "question": question,
                            "tool_call": tool_call
                        }
                    }, ensure_ascii=False)
            
            # Если вызова инструмента не было, логируем это
            logger.warning("Вызов инструмента не обнаружен в ответе LLM")
            content = response.content if hasattr(response, 'content') else str(response)
            logger.debug(f"Содержимое ответа LLM: {content}")
            
            return json.dumps({
                "answer": content,
                "success": True,
                "debug": {
                    "question": question,
                    "response": str(response)
                }
            }, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Agent error: {str(e)}", exc_info=True)
            return json.dumps({
                "error": str(e),
                "question": question,
            }, ensure_ascii=False)
            
    def execute_query(self, sql: str, params: tuple = None) -> List[Dict]:
        """Выполняет SQL-запрос в PostgreSQL"""
        conn = psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
        try:
            with conn.cursor() as cursor:
                if params:
                    logged_query = cursor.mogrify(sql, params).decode('utf-8')
                    logger.debug(f"Executing SQL query:\n{logged_query}")
                else:
                    logger.debug(f"Executing SQL query:\n{sql}")

                cursor.execute(sql, params or ())
                results = cursor.fetchall()
                logger.debug(f"Query returned {len(results)} results")
                return results
        except psycopg2.Error as e:
            logger.error(f"Database error: {str(e)}")
            return []
        finally:
            conn.close()

    def _execute_simple_search(self, species_name: str, search_type: str) -> str:
        """Реализация поиска с учетом типа контента (image/text)"""
        exact_match = species_name.lower()
        partial_match = f'%{species_name}%'
        alternative_match_1 = species_name.replace('ё', 'е')
        alternative_match_pattern_1 = f'%{alternative_match_1}%'
        alternative_match_2 = species_name.replace('е', 'ё')
        alternative_match_pattern_2 = f'%{alternative_match_2}%'
        params = (exact_match, partial_match, alternative_match_pattern_1, alternative_match_pattern_2)
        if search_type == 'image':
            sql = """
            SELECT 
                ri.file_path AS image_path,
                ri.name_ru AS name,
                r.access_options->>'original_title' AS description,
                'image' AS content_type
            FROM resources r
            JOIN resource_identifiers ri ON r.resource_id = ri.resource_id
            WHERE (ri.name_la ILIKE %s
                OR ri.name_ru ILIKE %s
                OR ri.name_ru ILIKE %s
                OR ri.name_ru ILIKE %s)
            AND r.content_type_id = (SELECT type_id FROM content_types WHERE name = 'изображение')
            LIMIT 5;
            """

        elif search_type == 'text':
            sql = """
            SELECT 
                r.content AS text_content,
                'text' AS content_type
            FROM resources r
            JOIN resource_identifiers ri ON r.resource_id = ri.resource_id
            WHERE (ri.name_la ILIKE %s
                OR ri.name_ru ILIKE %s
                OR ri.name_ru ILIKE %s
                OR ri.name_ru ILIKE %s)
            AND r.content_type_id = (SELECT type_id FROM content_types WHERE name = 'текст')
            LIMIT 1;
            """
        else:
            return json.dumps([{
                "error": f"Неподдерживаемый тип поиска: {search_type}",
                "suggestions": ["Используйте 'image' для изображений или 'text' для текста"]
            }], ensure_ascii=False)



        results = self.execute_query(sql, params)

        

        if not results:
            logger.warning(f"No results found for species: {species_name}")
            content_type_ru = "изображения" if search_type == "image" else "текстовое описание"
            return json.dumps([{
                "error": f"{content_type_ru.capitalize()} для вида '{species_name}' не найдены",
                "content_type": search_type,
                "suggestions": [
                    "Проверьте правильность написания названия",
                    "Попробуйте использовать латинское название",
                    "Уточните запрос, добавив дополнительные детали"
                ]
            }], ensure_ascii=False)

        return json.dumps(results, ensure_ascii=False)

    def _execute_attribute_search(self, species_name: str, attribute: str) -> str:
        """Реализация поиска с признаками"""
        sql = f"""
        SELECT 
            ri.file_path AS image_path,
            ri.name_ru AS name,
            r.access_options->>'original_title' AS description
        FROM resources r
        JOIN resource_identifiers ri ON r.resource_id = ri.resource_id
        WHERE (ri.name_la ILIKE '%{species_name}%'
               OR ri.name_ru ILIKE '%{species_name}%')
        AND r.content_type_id = (SELECT type_id FROM content_types WHERE name = 'изображение')
        AND (r.feature_data->>'description' ILIKE '%{attribute}%'
             OR r.access_options->>'original_title' ILIKE '%{attribute}%')
        LIMIT 5;
        """
        results = self.execute_query(sql)
        return json.dumps(results, ensure_ascii=False)

    def _execute_geo_search(self, species_name: str, location: str, distance_km: int) -> str:
        """Реализация геопоиска"""
        sql = f"""
        WITH location_point AS (
            SELECT ST_SetSRID(ST_MakePoint(104.3, 51.6), 4326)::geography AS geom
        )
        SELECT 
            sd.name AS location_name,
            sd.type AS geometry_type,
            ST_AsGeoJSON(sd.geometry)::json AS geojson,
            ST_Distance(sd.geometry, lp.geom) / 1000 AS distance_km
        FROM spatial_data sd, location_point lp
        JOIN resources r ON sd.resource_id = r.resource_id
        JOIN resource_identifiers ri ON r.resource_id = ri.resource_id
        WHERE (ri.name_la ILIKE '%{species_name}%' OR ri.name_ru ILIKE '%{species_name}%')
        AND ST_DWithin(sd.geometry, lp.geom, {distance_km * 1000})
        ORDER BY ST_Distance(sd.geometry, lp.geom)
        LIMIT 10;
        """
        results = self.execute_query(sql)
        return json.dumps(results, ensure_ascii=False)

    def _execute_text_description(self, species_name: str) -> str:
        """Реализация получения текстового описания"""
        sql = f"""
        SELECT 
            r.content
        FROM resources r
        JOIN resource_identifiers ri ON r.resource_id = ri.resource_id
        WHERE (ri.name_la ILIKE '%{species_name}%'
               OR ri.name_ru ILIKE '%{species_name}%')
        AND r.content_type_id = (SELECT type_id FROM content_types WHERE name = 'текст')
        LIMIT 1;
        """
        results = self.execute_query(sql)
        return json.dumps(results, ensure_ascii=False)

    def _get_agent_prompt(self) -> ChatPromptTemplate:
        """Возвращает промпт в правильном формате"""
        return ChatPromptTemplate.from_messages([
            ("system", (
                "Ты эксперт по Байкальской природной территории. "
                "Анализируй запросы пользователей и используй инструменты для ответа.\n\n"
                "Доступные инструменты:\n"
                "1. simple_search_tool - для поиска по названию вида\n"
                "Параметры:\n"
                "- species_name: название вида\n"
                "- search_type: 'image' или 'text'\n\n"
                "Правила выбора типа:\n"
                "- 'image' для запросов с: показать, фото, как выглядит\n"
                "- 'text' для запросов с: рассказать, описание, информация\n"
            )),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])

    def get_tools(self) -> list:
        """Возвращает список инструментов"""
        return self.tools
    def _get_agent_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", (
                "Ты эксперт по Байкальской природной территории. "
                "Твоя задача - анализировать запросы пользователей и ТОЧНО определять тип информации: "
                "'image' для изображений или 'text' для текста.\n\n"
                
                "ЖЕСТКИЕ ПРАВИЛА ВЫБОРА ТИПА:\n"
                "1. ВСЕГДА используй 'image' если в запросе есть:\n"
                "   - 'как выглядит', 'внешний вид', 'фото', 'изображение', 'снимок'\n"
                "   - 'покажи', 'показать', 'посмотреть', 'смотреть', 'найди картинку'\n"
                "   - 'фотография', 'картинка', 'рисунок', 'иллюстрация', 'визуально'\n"
                "2. Используй 'text' ТОЛЬКО если:\n"
                "   - Запрос содержит 'расскажи', 'описание', 'информация', 'текст', 'описать'\n"
                "   - И при этом НЕТ ключевых слов из пункта 1\n"
                "3. В ЛЮБЫХ сомнениях - выбирай 'image'\n\n"
                
                "КРИТИЧЕСКИ ВАЖНО:\n"
                "- Слова 'покажи', 'фото' и 'как выглядит' ВСЕГДА означают 'image'\n"
                "- Если есть противоречие (например 'покажи описание') - всё равно 'image'\n"
                "- Для простых названий видов без уточнений - ВСЕГДА 'image'\n\n"
                
                "НЕГАТИВНЫЕ ПРИМЕРЫ (КАК НЕ НАДО):\n"
                "- ❌ 'Покажи фото нерпы' → simple_search_tool('нерпа', 'text')\n"
                "- ❌ 'Как выглядит еж?' → simple_search_tool('ёж', 'text')\n\n"
                
                "ПРАВИЛЬНЫЕ ПРИМЕРЫ:\n"
                "- ✅ 'Покажи изображения даурского ежа' → simple_search_tool('даурский еж', 'image')\n"
                "- ✅ 'Фото байкальской нерпы' → simple_search_tool('байкальская нерпа', 'image')\n"
                "- ✅ 'Расскажи о жизни полевки' → simple_search_tool('полевка', 'text')\n"
                "- ✅ 'Даурский еж' → simple_search_tool('даурский еж', 'image')\n\n"
                
                "НИКОГДА не используй 'text' для запросов с: покажи/фото/как выглядит!"
            )),
        ("human", "{input}")  
    ])
sql_agent_instance = SQLAgent()
