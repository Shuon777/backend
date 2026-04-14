from sqlalchemy import Column, Integer, String, JSON, ForeignKey, Table
from sqlalchemy.orm import relationship
from .base import Base

object_synonym_link = Table(
    'object_name_synonym_link', Base.metadata,
    Column('object_id', Integer, ForeignKey('eco_assistant.object.id'), primary_key=True),
    Column('synonym_id', Integer, ForeignKey('eco_assistant.object_name_synonym.id'), primary_key=True),
    schema='eco_assistant'
)

class ObjectType(Base):
    __tablename__ = 'object_type'
    __table_args__ = {'schema': 'eco_assistant'}
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    schema = Column(JSON, default={})

class Object(Base):
    __tablename__ = 'object'
    __table_args__ = {'schema': 'eco_assistant'}
    id = Column(Integer, primary_key=True)
    db_id = Column(String, unique=True, nullable=False)
    object_type_id = Column(Integer, ForeignKey('eco_assistant.object_type.id'))
    object_properties = Column(JSON, default={})
    object_type = relationship('ObjectType')
    synonyms = relationship('ObjectNameSynonym', secondary=object_synonym_link)

class ObjectNameSynonym(Base):
    __tablename__ = 'object_name_synonym'
    __table_args__ = {'schema': 'eco_assistant'}
    id = Column(Integer, primary_key=True)
    synonym = Column(String, nullable=False)
    language = Column(String(10), default='ru')