import os
import shutil

from typing import Dict, Any, Tuple, List

from sqlalchemy import delete

from webob import Request, Response

from pibble.api.server.webservice.jsonapi import JSONWebServiceAPIServer
from pibble.api.exceptions import BadRequestError, NotFoundError, ConfigurationError
from pibble.ext.user.database import User
from pibble.ext.user.server.base import UserExtensionHandlerRegistry
from pibble.util.encryption import Password

from enfugue.api.controller.base import EnfugueAPIControllerBase

__all__ = ["EnfugueAPISystemController"]


def get_directory_size(directory: str, recurse: bool = True) -> Tuple[int, int, int]:
    """
    Sums the files and filesize of a directory
    """
    items = os.listdir(directory)
    top_level_items = len(items)
    files, size = 0, 0
    for item in items:
        path = os.path.join(directory, item)
        if os.path.isfile(path):
            files += 1
            size += os.path.getsize(path)
        elif recurse:
            sub_items, sub_files, sub_size = get_directory_size(path, recurse=True)
            files += sub_files
            size += sub_size
    return top_level_items, files, size


class EnfugueAPISystemController(EnfugueAPIControllerBase):
    handlers = UserExtensionHandlerRegistry()

    @handlers.path("^/api/settings$")
    @handlers.methods("GET")
    @handlers.format()
    @handlers.secured("System", "read")
    def get_settings(self, request: Request, response: Response) -> Dict[str, Any]:
        """
        Gets the settings that can be manipulated from the UI
        """
        return {
            "safe": self.configuration.get("enfugue.safe", True),
            "auth": not (self.configuration.get("enfugue.noauth", True)),
            "max_queued_invocations": self.manager.max_queued_invocations,
            "max_queued_downloads": self.manager.max_queued_downloads,
            "max_concurrent_downloads": self.manager.max_concurrent_downloads,
        }

    @handlers.path("^/api/settings$")
    @handlers.methods("POST")
    @handlers.format()
    @handlers.secured("System", "update")
    def update_settings(self, request: Request, response: Response) -> None:
        """
        Updates the settings that can be manipulated from the UI
        """
        if "auth" in request.parsed:
            self.user_config["enfugue.noauth"] = not request.parsed["auth"]
            if self.user_config["enfugue.noauth"] != request.parsed["auth"]:
                self.database.execute(delete(self.orm.AuthenticationToken))  # Clear auth data
                if request.parsed["auth"]:
                    self.database.execute(
                        delete(self.orm.User).filter(self.orm.User.username == "noauth")
                    )
                self.database.commit()
        if "safe" in request.parsed:
            if request.parsed["safe"] != self.configuration.get("enfugue.safe", True):
                # Destroy process so we re-initialize with new safety settings
                self.manager.stop_engine()
        for key in [
            "safe",
            "max_queued_invocation",
            "max_queued_downloads",
            "max_concurrent_downloads",
        ]:
            if key in request.parsed:
                self.user_config[f"enfugue.{key}"] = request.parsed[key]
        self.configuration.update(**self.user_config.dict())

    @handlers.path("^/api/users$")
    @handlers.methods("POST")
    @handlers.format()
    @handlers.secured("User", "create")
    def create_user(self, request: Request, response: Response) -> User:
        """
        Creates a user.
        """
        username = request.parsed.get("username", None)
        if not username:
            raise BadRequestError("Username is required.")
        user = (
            self.database.query(self.orm.User)
            .filter(self.orm.User.username == username)
            .one_or_none()
        )
        if user:
            raise BadRequestError(f"User {username} already exists.")
        password = request.parsed.get("new_password", None)
        repeat_password = request.parsed.get("repeat_password", None)

        if not password or not repeat_password:
            raise BadRequestError("Password is required.")
        if password != repeat_password:
            raise BadRequestError("Passwords do not match.")

        user = self.orm.User(username=username, password=Password.hash(password))

        if "first_name" in request.parsed:
            user.first_name = request.parsed["first_name"]
        if "last_name" in request.parsed:
            user.last_name = request.parsed["last_name"]

        self.database.add(user)
        self.database.commit()

        if "admin" in request.parsed and request.parsed["admin"]:
            admin_permission_group = (
                self.database.query(self.orm.PermissionGroup)
                .filter(self.orm.PermissionGroup.label == "admin")
                .one_or_none()
            )
            if not admin_permission_group:
                raise ConfigurationError(
                    "Couldn't find admin permission group. Did you modify the user initialization configuration?"
                )
            self.database.add(
                self.orm.UserPermissionGroup(user_id=user.id, group_id=admin_permission_group.id)
            )
        self.database.commit()
        return user

    @handlers.path("^/api/users/(?P<username>[a-zA-Z0-9_]+)$")
    @handlers.methods("PATCH")
    @handlers.format()
    @handlers.secured("User", "update")
    def update_user(self, request: Request, response: Response, username: str) -> User:
        """
        Updates one user.
        """
        user = (
            self.database.query(self.orm.User)
            .filter(self.orm.User.username == username)
            .one_or_none()
        )
        if not user:
            raise NotFoundError(f"No user named {username}")

        if "first_name" in request.parsed:
            user.first_name = request.parsed["first_name"]
        if "last_name" in request.parsed:
            user.last_name = request.parsed["last_name"]
        if "new_password" in request.parsed and "repeat_password" in request.parsed:
            if request.parsed["new_password"] != request.parsed["repeat_password"]:
                raise BadRequestError("Passwords do not match.")
            user.password = Password.hash(request.parsed["new_password"])
        if "admin" in request.parsed:
            if username == "enfugue" and not request.parsed["admin"]:
                raise BadRequestError("Cannot demote default user.")

            admin_permission_group = (
                self.database.query(self.orm.PermissionGroup)
                .filter(self.orm.PermissionGroup.label == "admin")
                .one_or_none()
            )
            if not admin_permission_group:
                raise ConfigurationError(
                    "Couldn't find admin permission group. Did you modify the user initialization configuration?"
                )
            admin_permission = None
            for user_permission_group in user.permission_groups:
                if user_permission_group.group_id == admin_permission_group.id:
                    admin_permission = user_permission_group
                    break

            if admin_permission is not None and not request.parsed["admin"]:
                self.database.delete(admin_permission)
            elif admin_permission is None and request.parsed["admin"]:
                self.database.add(
                    self.orm.UserPermissionGroup(
                        user_id=user.id, group_id=admin_permission_group.id
                    )
                )
        self.database.commit()
        return user

    @handlers.path("^/api/users/(?P<username>[a-zA-Z0-9_]+)$")
    @handlers.methods("DELETE")
    @handlers.format()
    @handlers.secured("User", "delete")
    def delete_user(self, request: Request, response: Response, username: str) -> None:
        """
        Deletes one user.
        We have to do the cascading ourselves because of a bug with sqlite and sqlalchemy.
        """
        if username == "enfugue":
            raise BadRequestError("Cannot delete default user.")
        user = (
            self.database.query(self.orm.User)
            .filter(self.orm.User.username == username)
            .one_or_none()
        )
        if not user:
            raise NotFoundError(f"No user named {username}")
        for permission in user.permissions:
            self.database.delete(permission)
        for permission_group in user.permission_groups:
            self.database.delete(permission_group)
        self.database.commit()
        self.database.delete(user)
        self.database.commit()

    @handlers.path("^/api/installation$")
    @handlers.methods("GET")
    @handlers.format()
    @handlers.secured("System", "read")
    def get_installation_summary(self, request: Request, response: Response) -> Dict[str, Any]:
        """
        Gets a summary of files and filesize in the installation
        """
        sizes = {}
        for dirname in ["cache", "checkpoint", "inversion", "lora", "models", "other"]:
            items, files, size = get_directory_size(os.path.join(self.engine_root, dirname))
            sizes[dirname] = {"items": items, "files": files, "bytes": size}
        return sizes

    @handlers.path("^/api/installation/(?P<dirname>[a-zA-Z0-9_]+)$")
    @handlers.methods("GET")
    @handlers.format()
    @handlers.secured("System", "read")
    def get_installation_details(
        self, request: Request, response: Response, dirname: str
    ) -> List[Dict[str, Any]]:
        """
        Gets a summary of files and filesize in the installation
        """
        directory = os.path.join(self.engine_root, dirname)
        if not os.path.isdir(directory):
            raise BadRequestError(f"Unknown engine directory {dirname}")

        items = []

        for item in os.listdir(directory):
            path = os.path.join(directory, item)
            if os.path.isdir(path):
                sub_items, files, size = get_directory_size(path)
                items.append({"type": "directory", "name": item, "bytes": size})
            else:
                items.append({"type": "file", "name": item, "bytes": os.path.getsize(path)})
        return items

    @handlers.path("^/api/installation/(?P<dirname>[^\/]+)/(?P<filename>[^\/]+)$")
    @handlers.methods("DELETE")
    @handlers.format()
    @handlers.secured("System", "update")
    def remove_from_installation(
        self, request: Request, response: Response, dirname: str, filename: str
    ) -> None:
        """
        Deletes a file or directory from the installation
        """

        path = os.path.join(self.engine_root, dirname, filename)
        if not os.path.exists(path):
            raise BadRequestError(f"Unknown engine file/directory {dirname}/{filename}")
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

    @handlers.bypass(JSONWebServiceAPIServer)
    @handlers.path("^/api/installation/(?P<dirname>[^\/]+)$")
    @handlers.methods("POST")
    @handlers.format()
    @handlers.secured("System", "update")
    def add_to_installation(self, request: Request, response: Response, dirname: str) -> None:
        """
        Uploads a file to an installation directory.
        """

        if "file" not in request.POST:
            raise BadRequestError("File is missing.")

        filename = request.POST["file"].filename
        dirpath = os.path.join(self.engine_root, dirname)
        if not os.path.exists(dirpath):
            raise BadRequestError(f"Unknonwn directory {dirname}")

        path = os.path.join(dirpath, filename)
        with open(path, "wb") as handle:
            for chunk in request.POST["file"].file:
                handle.write(chunk)
