from enfugue.database.base import EnfugueObjectBase
from sqlalchemy import Column, String, Integer, Float


class DiffusionModel(EnfugueObjectBase):
    __tablename__ = "model"

    name = Column(String(256), primary_key=True)

    model = Column(String(256), nullable=False)
    size = Column(Integer, nullable=False)
    prompt = Column(String(256), nullable=False, default="")
    negative_prompt = Column(String(256), nullable=False, default="")


class DiffusionModelLora(EnfugueObjectBase):
    __tablename__ = "model_lora"

    diffusion_model_name = Column(
        DiffusionModel.ForeignKey("name", ondelete="CASCADE", onupdate="CASCADE"), primary_key=True
    )
    model = Column(String(256), primary_key=True)
    weight = Column(Float, default=1.0, nullable=False)

    diffusion_model = DiffusionModel.Relationship(backref="lora")


class DiffusionModelInversion(EnfugueObjectBase):
    __tablename__ = "model_inversion"

    diffusion_model_name = Column(
        DiffusionModel.ForeignKey("name", ondelete="CASCADE", onupdate="CASCADE"), primary_key=True
    )
    model = Column(String(256), primary_key=True)

    diffusion_model = DiffusionModel.Relationship(backref="inversion")
