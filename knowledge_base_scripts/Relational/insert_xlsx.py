import pandas as pd
from sqlalchemy import create_engine, text
import json

df = pd.read_excel("OFF.xlsx")

engine = create_engine("postgresql://postgres:Fdf78yh0a4b!@testecobot.ru:5432/eco")

with engine.connect() as connection:
    for _, row in df.iterrows():
        if pd.isna(row['Русское']):
            continue
            
        latin_name = row['Латинское'] if not pd.isna(row['Латинское']) else None
        description = f"Вид растения: {row['Русское']}"

        result = connection.execute(
            text("""
            INSERT INTO biological_entity (common_name_ru, scientific_name, description, feature_data)
            VALUES (:common_name_ru, :scientific_name, :description, :feature_data)
            RETURNING id;
            """),
            {
                'common_name_ru': row['Русское'],
                'scientific_name': latin_name,
                'description': description,
                'feature_data': json.dumps({"classification": {"kingdom": "Plantae"}})
            }
        )
        bio_id = result.scalar()

        connection.execute(
            text("""
            INSERT INTO reliability (entity_table, entity_id, column_name, reliability_value)
            VALUES ('biological_entity', :entity_id, NULL, :reliability_value);
            """),
            {'entity_id': bio_id, 'reliability_value': 'научная'}
        )

        result = connection.execute(
            text("""
            INSERT INTO entity_identifier (name_ru, name_latin)
            VALUES (:name_ru, :name_latin)
            RETURNING id;
            """),
            {'name_ru': row['Русское'], 'name_latin': latin_name}
        )
        ident_id = result.scalar()

        connection.execute(
            text("""
            INSERT INTO entity_identifier_link (entity_id, entity_type, identifier_id)
            VALUES (:entity_id, 'biological_entity', :identifier_id);
            """),
            {'entity_id': bio_id, 'identifier_id': ident_id}
        )
        
    connection.commit()

print("Данные успешно импортированы в новую структуру БД!")