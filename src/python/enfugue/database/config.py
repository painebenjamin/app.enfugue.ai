from enfugue.database.base import EnfugueObjectBase
from pibble.database.orm import ORMVariadicType
from sqlalchemy import Column, String


class ConfigurationItem(EnfugueObjectBase):
    __tablename__ = "configuration_item"

    configuration_key = Column(String(256), primary_key=True)
    configuration_value = Column(ORMVariadicType(), nullable=True)
