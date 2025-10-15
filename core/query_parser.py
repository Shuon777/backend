from yargy import Parser, rule, or_
from yargy.pipelines import morph_pipeline
from yargy.interpretation import fact
import logging
from pymorphy3 import MorphAnalyzer
from functools import lru_cache

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@lru_cache(maxsize=None)
def get_morph_analyzer():
    return MorphAnalyzer()

morph = get_morph_analyzer()


class QueryModifier:
    def __init__(self):
        self.type_parser = self._create_info_type_parser()
        self.verb_parser = self._create_verb_parser()
        self._cache = {}
        self.habitat_parser = self._create_habitat_parser()
        self.appearance_parser = self._create_appearance_parser()
    def _create_appearance_parser(self):
        AppearancePhrase = fact('AppearancePhrase', ['type'])
        rules = [
            rule(
                morph_pipeline(['как']),
                morph_pipeline(['выглядит', 'выглядит?', 'выглядеть'])
            ).interpretation(AppearancePhrase.type.const('Изображение')),
            rule(
                morph_pipeline(['внешний']),
                morph_pipeline(['вид', 'облик'])
            ).interpretation(AppearancePhrase.type.const('Изображение'))
        ]
        return Parser(or_(*rules).interpretation(AppearancePhrase))

    def _normalize_word(self, word):
        if word not in self._cache:
            parsed = morph.parse(word)[0]
            self._cache[word] = parsed.normal_form
        return self._cache[word]


    def _create_info_type_parser(self):
        InfoType = fact('InfoType', ['type'])

        rules = [
            rule(morph_pipeline([
                'фото', 'изображение', 'картинка', 'рисунок',
                'снимок', 'иллюстрация', 'фотография', 'фотка'
            ])).interpretation(InfoType.type.const('Изображение')),

            rule(morph_pipeline([
                'видео', 'ролик', 'фильм', 'клип', 'кино',
                'мультимедиа', 'анимация'
            ])).interpretation(InfoType.type.const('Видео')),

            rule(morph_pipeline([
                'аудио', 'звук', 'подкаст', 'запись', 'озвучка',
                'музыка', 'шум', 'треки'
            ])).interpretation(InfoType.type.const('Аудио')),

            rule(morph_pipeline([
                'текст', 'описание', 'статья', 'информация',
                'сведения', 'контент', 'материал', 'документация', "справка"
            ])).interpretation(InfoType.type.const('Текст')),

            rule(morph_pipeline([
                'документ', 'файл', 'pdf', 'word', 'doc',
                'docx', 'презентация', 'excel'
            ])).interpretation(InfoType.type.const('Документ')),

            rule(morph_pipeline([
                'график', 'диаграмма', 'статистика', 'анализ данных',
                'отчет', 'визуализация', 'гистограмма', 'круговая'
            ])).interpretation(InfoType.type.const('Графики и диаграммы')),

            rule(morph_pipeline([
                'карта', 'гео', 'координаты', 'местность',
                'расположение', 'геопозиция', 'схема', 'план'
            ])).interpretation(InfoType.type.const('Картографическая информация')),

            rule(morph_pipeline([
                'трансляция', 'онлайн', 'стрим', 'прямой эфир',
                'live', 'вещание', 'репортаж'
            ])).interpretation(InfoType.type.const('Трансляция')),

            rule(morph_pipeline([
                'ссылка', 'ресурс', 'источник', 'сайт',
                'web', 'гиперссылка', 'url', 'портал'
            ])).interpretation(InfoType.type.const('Внешние ссылки'))
        ]

        return Parser(or_(*rules).interpretation(InfoType))

    def _create_verb_parser(self):
      Verb = fact('Verb', ['normalized'])
      verbs = [
          'покажи', 'показать', 'посмотреть', 'смотреть',
          'послушать', 'слушать', 'прочитать', 'читать',
          'скачать', 'открыть', 'перейти',
          'расскажи', 'рассказать', 'опиши', 'описать'
      ]
      return Parser(morph_pipeline(verbs).interpretation(Verb.normalized))

    def _create_habitat_parser(self):
        """Создает парсер для фраз, связанных с местом обитания"""
        HabitatPhrase = fact('HabitatPhrase', ['type'])

        rules = [
            rule(
                morph_pipeline(['покажи']),
                morph_pipeline(['где']),
                morph_pipeline(['обитает'])
            ).interpretation(HabitatPhrase.type.const('Картографическая информация')),
            rule(
                morph_pipeline(['покажи']),
                morph_pipeline(['где']),
                morph_pipeline(['растёт'])
            ).interpretation(HabitatPhrase.type.const('Картографическая информация')),
            
            rule(
                morph_pipeline(['покажи']),
                morph_pipeline(['ареал']),
                morph_pipeline(['обитания'])
            ).interpretation(HabitatPhrase.type.const('Картографическая информация')),

            rule(
                morph_pipeline(['где']),
                morph_pipeline(['можно']),
                morph_pipeline(['встретить'])
            ).interpretation(HabitatPhrase.type.const('Картографическая информация')),

            rule(
                morph_pipeline(['место']),
                morph_pipeline(['обитания'])
            ).interpretation(HabitatPhrase.type.const('Картографическая информация'))
        ]

        return Parser(or_(*rules).interpretation(HabitatPhrase))
    def modify(self, query: str) -> str:
        """Модифицирует запрос пользователя, добавляя информацию о типах запрашиваемых ресурсов.
        
        Args:
            query: Исходный запрос пользователя
            
        Returns:
            Модифицированный запрос с добавленным type:, если найдены соответствующие паттерны
        """
        found_types = set()

        appearance_matches = list(self.appearance_parser.findall(query))
        for match in appearance_matches:
            if hasattr(match.fact, 'type'):
                return f"{query} type:{match.fact.type}"

        habitat_matches = list(self.habitat_parser.findall(query))
        for match in habitat_matches:
            if hasattr(match.fact, 'type'):
                return f"{query} type:{match.fact.type}"

        type_matches = list(self.type_parser.findall(query))
        for match in type_matches:
            if hasattr(match.fact, 'type'):
                found_types.add(match.fact.type)

        verb_mapping = {
            'покажи': 'Изображение',
            'показать': 'Изображение',
            'посмотреть': 'Видео',
            'смотреть': 'Видео',
            'послушать': 'Аудио',
            'слушать': 'Аудио',
            'прочитать': 'Текст',
            'читать': 'Текст',
            'рассказать': 'Текст',
            'скачать': 'Документ',
            'открыть': 'Внешние ссылки',
            'перейти': 'Внешние ссылки'
        }

        for match in self.verb_parser.findall(query):
            if match.tokens:
                token = match.tokens[0]
                if token.forms:
                    normal_verb = token.forms[0].normalized
                    if normal_verb in verb_mapping:
                        found_types.add(verb_mapping[normal_verb])

        if found_types:
            unique_types = {t for t in found_types if t}
            if unique_types:
                return f"{query} type:{'|'.join(sorted(unique_types))}"

        return query
