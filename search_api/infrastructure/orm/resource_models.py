from sqlalchemy import Column, Integer, String, JSON, ForeignKey, Date, Table
from sqlalchemy.orm import relationship
from .base import Base

class Author(Base):
    __tablename__ = 'author'
    __table_args__ = {'schema': 'eco_assistant'}
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)

class Source(Base):
    __tablename__ = 'source'
    __table_args__ = {'schema': 'eco_assistant'}
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)

class ReliabilityLevel(Base):
    __tablename__ = 'reliability_level'
    __table_args__ = {'schema': 'eco_assistant'}
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)

class Bibliographic(Base):
    __tablename__ = 'bibliographic'
    __table_args__ = {'schema': 'eco_assistant'}
    id = Column(Integer, primary_key=True)
    author_id = Column(Integer, ForeignKey('eco_assistant.author.id'))
    date = Column(Date)
    source_id = Column(Integer, ForeignKey('eco_assistant.source.id'))
    reliability_level_id = Column(Integer, ForeignKey('eco_assistant.reliability_level.id'))
    
    author = relationship('Author', backref='bibliographic_records')
    source = relationship('Source', backref='bibliographic_records')
    reliability_level = relationship('ReliabilityLevel', backref='bibliographic_records')

class Creation(Base):
    __tablename__ = 'creation'
    __table_args__ = {'schema': 'eco_assistant'}
    id = Column(Integer, primary_key=True)
    creation_type = Column(String)
    creation_tool = Column(String)
    creation_params = Column(JSON)

class ResourceStatic(Base):
    __tablename__ = 'resource_static'
    __table_args__ = {'schema': 'eco_assistant'}
    id = Column(Integer, primary_key=True)
    static_id = Column(String, unique=True)
    bibliographic_id = Column(Integer, ForeignKey('eco_assistant.bibliographic.id'))
    creation_id = Column(Integer, ForeignKey('eco_assistant.creation.id'))
    
    bibliographic = relationship('Bibliographic', backref='resource_statics')
    creation = relationship('Creation', backref='resource_statics')

class SupportMetadata(Base):
    __tablename__ = 'support_metadata'
    __table_args__ = {'schema': 'eco_assistant'}
    id = Column(Integer, primary_key=True)
    parameters = Column(JSON, nullable=False)

class Resource(Base):
    __tablename__ = 'resource'
    __table_args__ = {'schema': 'eco_assistant'}
    id = Column(Integer, primary_key=True)
    title = Column(String)
    uri = Column(String)
    features = Column(JSON)
    text_id = Column(String, unique=True)
    resource_static_id = Column(Integer, ForeignKey('eco_assistant.resource_static.id'))
    support_metadata_id = Column(Integer, ForeignKey('eco_assistant.support_metadata.id'))
    
    resource_static = relationship('ResourceStatic', backref='resources')
    support_metadata = relationship('SupportMetadata', backref='resources')
    objects = relationship('Object', secondary='eco_assistant.resource_object', backref='resources')

resource_object_table = Table(
    'resource_object', Base.metadata,
    Column('resource_id', Integer, ForeignKey('eco_assistant.resource.id'), primary_key=True),
    Column('object_id', Integer, ForeignKey('eco_assistant.object.id'), primary_key=True),
    Column('relation_type', String),
    schema='eco_assistant'
)