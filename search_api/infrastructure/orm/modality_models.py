from sqlalchemy import Column, Integer, String, JSON, ForeignKey
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from .base import Base

class Modality(Base):
    __tablename__ = 'modality'
    __table_args__ = {'schema': 'eco_assistant'}
    id = Column(Integer, primary_key=True)
    modality_type = Column(String, unique=True, nullable=False)
    value_table_name = Column(String, nullable=False)

class TextValue(Base):
    __tablename__ = 'text_value'
    __table_args__ = {'schema': 'eco_assistant'}
    id = Column(Integer, primary_key=True)
    structured_data = Column(JSON, nullable=False)

class ImageValue(Base):
    __tablename__ = 'image_value'
    __table_args__ = {'schema': 'eco_assistant'}
    id = Column(Integer, primary_key=True)
    url = Column(String)
    file_path = Column(String)
    format = Column(String(20))

class GeodataValue(Base):
    __tablename__ = 'geodata_value'
    __table_args__ = {'schema': 'eco_assistant'}
    id = Column(Integer, primary_key=True)
    geometry = Column(Geometry(geometry_type='GEOMETRY', srid=4326), nullable=False)
    geometry_type = Column(String)

class ResourceValue(Base):
    __tablename__ = 'resource_value'
    __table_args__ = {'schema': 'eco_assistant'}
    id = Column(Integer, primary_key=True)
    resource_id = Column(Integer, ForeignKey('eco_assistant.resource.id'))
    modality_id = Column(Integer, ForeignKey('eco_assistant.modality.id'))
    value_id = Column(Integer)
    
    modality = relationship('Modality', backref='resource_values')
    resource = relationship('Resource', backref='resource_values')