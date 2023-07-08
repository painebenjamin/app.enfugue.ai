from enfugue.database.base import EnfugueObjectBase
from pibble.database.orm import ORMVariadicType
from sqlalchemy import Column, String, Integer, Float

__all__ = [
    "DiffusionModel",
    "DiffusionModelRefiner",
    "DiffusionModelDefaultConfiguration",
    "DiffusionModelLora",
    "DiffusionModelLycoris",
    "DiffusionModelInversion"
]

class DiffusionModel(EnfugueObjectBase):
    __tablename__ = "model"

    name = Column(String(256), primary_key=True)

    model = Column(String(256), nullable=False)
    size = Column(Integer, nullable=False)
    prompt = Column(String(256), nullable=False, default="")
    negative_prompt = Column(String(256), nullable=False, default="")

class DiffusionModelRefiner(EnfugueObjectBase):
    __tablename__ = "model_refiner"
    
    diffusion_model_name = Column(
        DiffusionModel.ForeignKey("name", ondelete="CASCADE", onupdate="CASCADE"), primary_key=True, unique=True
    )
    model = Column(String(256), nullable=False)

    diffusion_model = DiffusionModel.Relationship(backref="refiner", uselist=False)

class DiffusionModelDefaultConfiguration(EnfugueObjectBase):
    __tablename__ = "model_default_configuration"
    
    diffusion_model_name = Column(
        DiffusionModel.ForeignKey("name", ondelete="CASCADE", onupdate="CASCADE"), primary_key=True
    )
    
    configuration_key = Column(String(256), primary_key=True)
    configuration_value = Column(ORMVariadicType(), nullable=True)

    diffusion_model = DiffusionModel.Relationship(backref="config")

class DiffusionModelLora(EnfugueObjectBase):
    __tablename__ = "model_lora"

    diffusion_model_name = Column(
        DiffusionModel.ForeignKey("name", ondelete="CASCADE", onupdate="CASCADE"), primary_key=True
    )
    model = Column(String(256), primary_key=True)
    weight = Column(Float, default=1.0, nullable=False)

    diffusion_model = DiffusionModel.Relationship(backref="lora")

class DiffusionModelLycoris(EnfugueObjectBase):
    __tablename__ = "model_lycoris"

    diffusion_model_name = Column(
        DiffusionModel.ForeignKey("name", ondelete="CASCADE", onupdate="CASCADE"), primary_key=True
    )
    model = Column(String(256), primary_key=True)
    weight = Column(Float, default=1.0, nullable=False)

    diffusion_model = DiffusionModel.Relationship(backref="lycoris")

class DiffusionModelInversion(EnfugueObjectBase):
    __tablename__ = "model_inversion"

    diffusion_model_name = Column(
        DiffusionModel.ForeignKey("name", ondelete="CASCADE", onupdate="CASCADE"), primary_key=True
    )
    model = Column(String(256), primary_key=True)

    diffusion_model = DiffusionModel.Relationship(backref="inversion")
