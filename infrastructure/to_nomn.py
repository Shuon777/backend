import pymorphy2
from infrastructure.geo_db_store import load_db
morph = pymorphy2.MorphAnalyzer()

def to_prepositional_phrase(text):
    words = text.split()
    parsed_words = [morph.parse(w)[0] for w in words]

    # Найдём первое существительное
    for i, pw in enumerate(parsed_words):
        if 'NOUN' in pw.tag:
            noun_index = i
            break
    else:
        return text  # нет существительного — не обрабатываем

    # Определим род, число, падеж (предложный)
    noun = parsed_words[noun_index]
    gender = noun.tag.gender
    number = noun.tag.number
    case = 'nomn'  # предложный падеж

    # Преобразуем все слова с учётом согласования
    result = []
    for i, pw in enumerate(parsed_words):
        if i == noun_index:
            inflected = pw.inflect({case})
        elif i < noun_index and ('ADJF' in pw.tag or 'PRTF' in pw.tag):  # прилагательное перед существительным
            inflected = pw.inflect({case, gender, number})
        else:
            inflected = pw.inflect({case})
        result.append(inflected.word if inflected else pw.word)

    return " ".join(result)

def normalize_text(text: str) -> str:
    return " ".join([morph.parse(w)[0].normal_form for w in text.strip().split()]).lower()

def find_place_key(user_input):
    db = load_db()
    input_normalized = normalize_text(user_input)
    for key in db:
        if normalize_text(key) == input_normalized:
            return key
    return None