from enfugue.database.base import EnfugueObjectBase
from pibble.database.orm import ORMVariadicType
from pibble.ext.user.database import User
from sqlalchemy import Column, String, DateTime, Integer
from sqlalchemy.sql import func


class DiffusionInvocation(EnfugueObjectBase):
    __tablename__ = "invocation"

    id = Column(String(32), primary_key=True)
    user_id = Column(User.ForeignKey("id", ondelete="SET NULL", onupdate="SET NULL"), nullable=True)
    started = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    plan = Column(ORMVariadicType(), nullable=False)
    duration = Column(Integer(), nullable=False, default=0)
    outputs = Column(Integer(), nullable=False, default=0)
    error = Column(String(512), nullable=True)

    user = User.Relationship(backref="invocations")
