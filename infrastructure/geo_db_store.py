"""
Модуль для простой файловой базы данных (JSON) по географическим объектам.

Используется для кеширования результатов поиска, включая геометрию и ссылки на карты.
"""
import json
import os
import difflib
from typing import Any, Dict, Optional
import re

import pymorphy2
morph = pymorphy2.MorphAnalyzer()

def normalize_morph(text: str) -> str:
    return ' '.join([morph.parse(word)[0].normal_form for word in text.lower().split()])

DB_PATH = "json_files/geodb.json"

if not os.path.exists(DB_PATH):
    with open(DB_PATH, "w") as f:
        json.dump({}, f)

def load_db() -> Dict[str, Any]:
    """Загружает базу данных из файла JSON."""
    with open(DB_PATH, "r") as f:
        return json.load(f)

def save_db(data: Dict[str, Any]) -> None:
    """Сохраняет словарь в файл базы данных JSON."""
    with open(DB_PATH, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_place(name: str) -> Optional[Dict[str, Any]]:
    """Возвращает данные по месту, если оно есть в базе."""
    db = load_db()
    return db.get(name.lower())

def add_place(name: str, data: Dict[str, Any]) -> None:
    """Добавляет или обновляет данные по месту в базе."""
    db = load_db()
    db[name.lower()] = data
    save_db(db)

def normalize_lower(text: str) -> str:
    return text.strip().lower()

def _tokens(s: str) -> set[str]:
    # слова ≥3 символов, в нижнем регистре
    return set(re.findall(r"[а-яёa-z]{3,}", s.lower()))

def find_place_flexible(user_input: str, cutoff: float = 0.80) -> dict:
    db = load_db()
    user_norm = normalize_morph(user_input)

    # индекс: нормализованный ключ -> исходный ключ
    norm2orig = {normalize_morph(k): k for k in db.keys()}

    # 1) точное совпадение в нормализованном пространстве
    if user_norm in norm2orig:
        orig = norm2orig[user_norm]
        return {"name": orig, "record": db[orig]}

    # 2) "подстрока", но по словам и с порогом покрытия
    u_tokens = _tokens(user_norm)
    best = None
    best_score = 0.0

    for nk, orig in norm2orig.items():
        c_tokens = _tokens(nk)
        inter = len(u_tokens & c_tokens)
        if inter == 0:
            continue
        # доля общих слов относительно большего набора
        score = inter / max(len(u_tokens), len(c_tokens))
        if score >= 0.8 and score > best_score:
            best = orig
            best_score = score

    if best:
        return {"name": best, "record": db[best]}

    # 3) fuzzy только по НОРМАЛИЗОВАННЫМ ключам
    keys_norm = list(norm2orig.keys())
    close = difflib.get_close_matches(user_norm, keys_norm, n=1, cutoff=cutoff)
    if close:
        orig = norm2orig[close[0]]
        return {"name": orig, "record": db[orig]}

    return {"name": "not_found", "record": "not_found"}
