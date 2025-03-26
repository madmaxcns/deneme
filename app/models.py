from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base

from pydantic import BaseModel

from pydantic import BaseModel

class HastaMetinCreate(BaseModel):
    hasta_no: int
    paragraf: str
    cinsiyet: str

class HastaSorularCreate(BaseModel):
    hasta_no: int
    soru_1: int
    soru_2: int
    soru_3: int
    soru_4: int
    soru_5: int
    soru_6: int
    soru_7: int
    soru_8: int
    soru_9: int
    soru_10: int
    soru_11: int
    soru_12: int
    soru_13: int
    soru_14: int
    soru_15: int
    soru_16: int
    soru_17: int
    soru_18: int
    soru_19: int
    soru_20: int

class HastaGozlemCreate(BaseModel):
    hasta_no: int
    paragraf_sonuc: str
    test_sonuc: str




Base = declarative_base()

class HastaMetin(Base):
    __tablename__ = 'hasta_metin'
    sira_numarasi = Column(Integer, primary_key=True, autoincrement=True)
    hasta_no = Column(Integer, index=True)
    paragraf = Column(Text)
    cinsiyet = Column(String)

class HastaSorular(Base):
    __tablename__ = 'hasta_sorular'
    sira_numarasi = Column(Integer, primary_key=True)
    hasta_no = Column(Integer)
    soru_1 = Column(Integer)
    soru_2 = Column(Integer)
    soru_3 = Column(Integer)
    soru_4 = Column(Integer)
    soru_5 = Column(Integer)
    soru_6 = Column(Integer)
    soru_7 = Column(Integer)
    soru_8 = Column(Integer)
    soru_9 = Column(Integer)
    soru_10 = Column(Integer)
    soru_11 = Column(Integer)
    soru_12 = Column(Integer)
    soru_13 = Column(Integer)
    soru_14 = Column(Integer)
    soru_15 = Column(Integer)
    soru_16 = Column(Integer)
    soru_17 = Column(Integer)
    soru_18 = Column(Integer)
    soru_19 = Column(Integer)
    soru_20 = Column(Integer)
    soru_20 = Column(Integer)

class HastaGozlem(Base):
    __tablename__ = 'hasta_gozlem'
    sira_numarasi = Column(Integer, primary_key=True, autoincrement=True)
    hasta_no = Column(Integer, index=True)
    paragraf_sonuc = Column(String(50))
    test_sonuc = Column(String(50))
