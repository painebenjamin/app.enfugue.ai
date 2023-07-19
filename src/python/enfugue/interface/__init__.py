import os
import io
import enfugue

from webob import Request, Response

from typing import Any, Union, Dict

from lxml import etree as ET
from lxml.builder import E

from pibble.util.helpers import url_join
from pibble.util.imaging import contrast_color
from pibble.util.strings import snake_case, Serializer

from pibble.api.exceptions import NotFoundError
from pibble.api.server.webservice.template import TemplateServer

from pibble.api.middleware.database.orm import ORMMiddlewareBase

from pibble.ext.cms.middleware import CMSExtensionContextMiddleware
from pibble.ext.cms.server.extension import (
    CMSExtensionStaticExtension,
    CMSExtensionResolveStatementExtension,
    CMSExtensionResolveFunctionExtension,
)
from pibble.ext.user.server.base import (
    UserExtensionTemplateServer,
    UserExtensionTemplateHandlerRegistry,
    UserExtensionServerBase,
)
from pibble.ext.session.server.base import SessionExtensionServerBase

from enfugue.util import logger, get_local_static_directory
from enfugue.interface.helpers import (
    HTMLPropertiesHelperFunction,
    SerializeHelperFunction,
    SerializeHelperFilter,
    CheckResolveURLHelperFilter,
)
from enfugue.database import *


class EnfugueInterfaceServer(
    UserExtensionTemplateServer,
    CMSExtensionContextMiddleware,
    SessionExtensionServerBase,
):
    handlers = UserExtensionTemplateHandlerRegistry()

    def configure(self, **configuration: Any) -> None:
        """
        Intercept configure() to add the static directory.
        """
        production_root = get_local_static_directory()
        production_templates = os.path.join(production_root, "html")

        development_root = os.path.abspath(os.path.join(os.path.dirname(enfugue.__file__), "../../../src"))
        development_templates = os.path.join(development_root, "html")

        static_dirs = [production_root]
        template_dirs = []

        if os.path.exists(development_root):
            static_dirs.append(development_root)
            if os.path.exists(development_templates):
                template_dirs.append(development_templates)
        if os.path.exists(production_templates):
            template_dirs.append(production_templates)

        server = configuration.get("server", {})
        static_config = server.get("static", {})
        static_directories = static_config.get("directories", [])
        static_directories.extend(static_dirs)
        static_config["directories"] = static_directories
        server["static"] = static_config

        template_config = server.get("template", {})
        template_directories = template_config.get("directories", [])
        template_directories.extend(template_dirs)
        template_config["directories"] = template_directories
        server["template"] = template_config

        configuration["server"] = server
        super(EnfugueInterfaceServer, self).configure(**configuration)

    def on_configure(self):
        """
        The application interface extends the entire Enfugue database structure,
        with the exception of admin objects.
        """
        self.orm.extend_base(EnfugueObjectBase)
        self.templates.extend(
            CMSExtensionStaticExtension,
            CMSExtensionResolveStatementExtension,
            CMSExtensionResolveFunctionExtension,
            HTMLPropertiesHelperFunction,
            CheckResolveURLHelperFilter,
            SerializeHelperFunction,
            SerializeHelperFilter,
        )

    def prepare_context(self, request: Request, response: Response) -> Dict[str, Any]:
        """
        This is the function called by the TemplateServer to get the context available
        in templates.
        """
        context = {
            "current_path": request.path,
            "current_url": request.url,
            "canonical_url": url_join(self.configuration["server.cms.path.root"], request.path),
            "orm": self.orm,
            "database": self.database,
            "paths": self.configuration["server.cms.path"],
        }

        if hasattr(request, "token"):
            context["user"] = request.token.user
            context["admin"] = self.check_user_permission(request.token.user, "System", "update")

        return context

    @handlers.path("^/$")
    @handlers.methods("GET")
    @handlers.template("content-application.html.j2")
    @handlers.reverse("Application", "/")
    @handlers.secured()
    def root(self, request: Request, response: Response) -> Dict[str, Any]:
        """
        At the root, we display the application.

        If the user isn't logged in, they'll see a login there.
        """
        return {"title": "Application"}

    @handlers.bypass(
        SessionExtensionServerBase,
        UserExtensionTemplateServer,
        TemplateServer,
        UserExtensionServerBase,
        ORMMiddlewareBase,
    )  # Bypass token checking and ORM instantiation
    @handlers.methods("GET")
    @handlers.cache(86400)
    @handlers.path("^/manifest.json$")
    def get_manifest(self, request: Request, response: Response) -> None:
        """
        Gets the browser manifest.
        """
        app_name = self.configuration.get("server.cms.name", "Pibble App Template")
        app_color = self.configuration.get("server.cms.context.base.meta.theme-color", "#ffffff")

        response.content_type = "application/json"
        response.json = {
            "name": app_name,
            "short_name": self.configuration.get("server.cms.short_name", snake_case(app_name)),
            "icons": [
                {
                    "src": url_join(
                        self.configuration["server.cms.path.root"],
                        self.configuration["server.cms.context.base.android.icon.href"],
                    ),
                    "sizes": self.configuration["server.cms.context.base.android.icon.size"],
                    "type": self.configuration["server.cms.context.base.android.icon.type"],
                }
            ],
            "theme_color": self.configuration.get("server.cms.context.base.meta.theme-color", "#ffffff"),
            "background_color": "#000000" if contrast_color(app_color) == "black" else "#ffffff",
            "display": "standalone",
        }

    @handlers.bypass(
        SessionExtensionServerBase,
        UserExtensionTemplateServer,
        TemplateServer,
        UserExtensionServerBase,
        ORMMiddlewareBase,
    )  # Bypass token checking and ORM instantiation
    @handlers.methods("GET")
    @handlers.cache(86400)
    @handlers.path("^/browserconfig.xml$")
    def get_browserconfig(self, request: Request, response: Response) -> None:
        """
        Gets the 'browserconfig' - an xml config for windows shortcuts.
        """
        app_color = self.configuration.get("server.cms.context.base.meta.theme-color", "#ffffff")
        response.content_type = "application/xml"
        response.body = ET.tostring(
            E.browserconfig(
                E.msapplication(
                    E.tile(
                        E(
                            "square{0}logo".format(self.configuration["server.cms.context.base.windows.icon.size"]),
                            src=url_join(
                                self.configuration["server.cms.path.root"],
                                self.configuration["server.cms.context.base.windows.icon.href"],
                            ),
                        ),
                        E.TileColor(app_color),
                    )
                )
            ),
            xml_declaration=True,
            encoding="utf-8",
        )

    @handlers.bypass(
        SessionExtensionServerBase,
        UserExtensionTemplateServer,
        TemplateServer,
        UserExtensionServerBase,
        ORMMiddlewareBase,
    )  # Bypass token checking and ORM instantiation
    @handlers.methods("GET")
    @handlers.cache(86400)
    @handlers.path("^/favicon.ico$")
    def get_root_favicon(self, request: Request, response: Response) -> None:
        """
        Gets the root favicon, which browsers will sometimes request before
        the page has completed loading (which would have told them the .ico is
        on the static path)
        """
        self.redirect(
            response,
            url_join(
                self.configuration["server.cms.path.root"],
                self.configuration["server.cms.context.base.links"]["shortcut icon"][0]["href"],
            ),
        )

    @handlers.bypass(
        SessionExtensionServerBase,
        UserExtensionTemplateServer,
        TemplateServer,
        UserExtensionServerBase,
        ORMMiddlewareBase,
    )  # Bypass token checking and ORM instantiation
    @handlers.reverse("static", "/static/{path}")
    @handlers.methods("GET")
    @handlers.path("/static/(?P<path>\S+)")
    @handlers.cache()
    @handlers.download()
    def static(self, request: Request, response: Response, path: str) -> Union[str, io.BytesIO, io.BufferedReader]:
        directories = self.configuration.get("server.static.directories", [])
        if path.endswith("/index.autogenerated.mjs"):
            # Create autogenerated file index for JS directories.
            for root in directories:
                dir_path = os.path.dirname(os.path.join(root, path))
                if os.path.exists(dir_path) and os.path.isdir(dir_path):
                    js_files = [filename for filename in os.listdir(dir_path) if filename.endswith("js")]
                    if len(js_files) > 0:
                        index = "const Index = {0}; export {{ Index }};".format(
                            Serializer.serialize(list(sorted(js_files)))
                        )
                        response.content_type = "application/javascript"
                        return io.BytesIO(index.encode("utf-8"))
            raise NotFoundError(f"No javascript files found at path {path[:-23]}")
        for root in directories:
            root_path = os.path.join(root, path)
            if os.path.exists(root_path) and os.path.isfile(root_path):
                if root_path.endswith(".mjs"):
                    # Force content-type
                    response.content_type = "application/javascript"
                    return open(root_path, "rb")
                else:
                    # Let pibble infer by filename
                    return root_path
        logger.debug(f"Couldn't find path {path}, tried {directories}")
        raise NotFoundError(f"No file at {path}")

    @handlers.errors(401)
    def handle_authentication_error(self, request: Request, response: Response) -> None:
        """
        Determine what to do when authentication errors occur, then do it.
        """
        if self.configuration.get("enfugue.noauth", False):
            # Bypass login
            self.bypass_login(request, response)
            self.handle_request(request, response)  # type: ignore
        else:
            self.redirect(response, "/login", 302)

    @handlers.cache(86400)
    @handlers.template("content-login.html.j2")
    @handlers.methods("GET")
    @handlers.path("^/login$")
    def view_login(self, request: Request, response: Response) -> Dict[str, Any]:
        """
        The GET() handler for logging in, i.e, the landing page when a user isn't logged in.
        """
        if self.configuration.get("enfugue.noauth", False):
            if not hasattr(request, "token"):
                self.bypass_login(request, response)
            self.redirect(response, "/", 302)
        return {"title": "Login"}

    @handlers.methods("POST")
    @handlers.template("content-login.html.j2")
    @handlers.path("^/login$")
    @handlers.reverse("Login", "/login")
    def process_login(self, request, response) -> Dict[str, Any]:
        """
        The POST() handler for logging in.
        """
        context = {"title": "Login"}
        try:
            token = self.login(request, response)
        except Exception as ex:
            context["error"] = str(ex)
            context["username"] = request.POST["username"]
        else:
            self.redirect(
                response,
                request.POST["redirect"]
                if "redirect" in request.POST
                else url_join(
                    self.configuration["server.cms.path.root"],
                    self.resolve("Application", controller="data"),
                ),
            )
        return context

    @handlers.secured()
    @handlers.methods("GET")
    @handlers.reverse("Logout", "/logout")
    @handlers.path("^/logout$")
    def process_logout(self, request: Request, response: Response) -> None:
        """
        Logout() is a GET() handler that just calls the UserExtension parent method.
        """
        context = {"title": "Logout"}
        self.logout(request, response)
        self.redirect(
            response,
            url_join(self.configuration["server.cms.path.root"], self.resolve("Login")),
        )
