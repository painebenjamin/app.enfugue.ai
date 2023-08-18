from typing import Any, Tuple, Iterable, Dict
from pibble.database.orm import ORM

__all__ = ["EnfugueConfiguration"]


class NoDefault:
    pass


class EnfugueConfiguration:
    """
    This class allows us to get and set items to the database using a dict syntax
    """

    def __init__(self, orm: ORM) -> None:
        self.orm = orm

    def get(self, key: str, default: Any = NoDefault) -> Any:
        """
        Gets an item, allow a default
        """
        try:
            return self[key]
        except KeyError:
            if default is NoDefault:
                raise
            return default

    def items(self) -> Iterable[Tuple[str, Any]]:
        """
        Iterates over the entire set of key-value pairs.
        """
        with self.orm.session() as session:
            for item in session.query(self.orm.ConfigurationItem).all():
                yield (item.configuration_key, item.configuration_value)

    def dict(self) -> Dict[str, Any]:
        """
        Gets all items as a dict.
        """
        return dict([item for item in self.items()])

    def __delitem__(self, key: str) -> None:
        """
        Deletes an item from the database
        """
        with self.orm.session() as session:
            item = (
                session.query(self.orm.ConfigurationItem)
                .filter(self.orm.ConfigurationItem.configuration_key == key)
                .one_or_none()
            )
            if item:
                session.delete(item)
                session.commit()

    def __getitem__(self, key: str) -> Any:
        """
        Gets an item from the database
        """
        with self.orm.session() as session:
            item = (
                session.query(self.orm.ConfigurationItem)
                .filter(self.orm.ConfigurationItem.configuration_key == key)
                .one_or_none()
            )
            if item is None:
                raise KeyError(key)
            return item.configuration_value

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Sets one item in the database
        """
        with self.orm.session() as session:
            item = (
                session.query(self.orm.ConfigurationItem)
                .filter(self.orm.ConfigurationItem.configuration_key == key)
                .one_or_none()
            )
            if item is None:
                session.add(self.orm.ConfigurationItem(configuration_key=key, configuration_value=value))
            else:
                item.configuration_value = value
            session.commit()
