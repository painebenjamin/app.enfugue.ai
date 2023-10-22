from __future__ import annotations

from typing import Optional, Any, Dict, Iterator, Literal, Any

from html2text import HTML2Text
from pibble.api.client.webservice.jsonapi import JSONWebServiceAPIClient
from pibble.util.strings import Serializer

__all__ = ["CivitAIModel", "CivitAI"]


class CivitAIModel:
    """
    This represents an individual result from CivitAI
    """

    parser: HTML2Text

    @property
    def html_parser(self) -> HTML2Text:
        """
        Gets the parser that turns HTML to plaintext.
        """
        if not hasattr(CivitAIModel, "parser"):
            CivitAIModel.parser = HTML2Text()
            CivitAIModel.parser.ignore_links = True
            CivitAIModel.parser.ignore_emphasis = True
            CivitAIModel.parser.bypass_tables = True
        return CivitAIModel.parser

    def __init__(self, **properties: Any) -> None:
        """
        On initialization, clean description.
        """
        self.properties = properties
        if self.properties.get("description", None) is not None:
            lines = self.html_parser.handle(self.properties["description"]).splitlines()
            self.properties["description"] = "\n".join([line.strip() for line in lines if line.strip()])

    def serialize(self) -> Dict[str, Any]:
        """
        When serializing just return the properties dict.
        """
        return self.properties

    def __str__(self) -> str:
        """
        When stringifying, run through the serializer.
        """
        return Serializer.serialize(self.serialize())


class CivitAI(JSONWebServiceAPIClient):
    """
    This class provides some easy abstractions for querying from CivitAI.
    """

    PAGE_SIZE = 100

    def configure(self, client: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        if client is None:
            client = {}
        client["secure"] = True
        client["host"] = "civitai.com"
        client["path"] = "/api/v1"
        super(CivitAI, self).configure(client=client, **kwargs)

    def get_all(self, path: str, limit: Optional[int] = None, **parameters: Any) -> Iterator[Dict[str, Any]]:
        """
        Gets all results from an endpoint, paginating if necessary
        """

        parameters = dict([(key, value) for key, value in parameters.items() if value is not None and value != ""])

        if limit is not None:
            parameters["limit"] = min(limit, self.PAGE_SIZE)

        # Get first page
        page = self.get(path, parameters=parameters).json()
        total_items = 0

        while page is not None:
            for item in page["items"]:
                total_items += 1
                yield item
                if limit is not None and total_items >= limit:
                    return
            if page["metadata"].get("nextPage", None) is not None:
                # Remove limit, it gets sent in the next page URL
                page = self.get(page["metadata"]["nextPage"]).json()
            else:
                page = None

    def get_models(
        self,
        limit: Optional[int] = None,
        page: Optional[int] = None,
        query: Optional[str] = None,
        tag: Optional[str] = None,
        username: Optional[str] = None,
        types: Optional[
            Literal[
                "Checkpoint",
                "TextualInversion",
                "Hypernetwork",
                "AestheticGradient",
                "LORA",
                "Controlnet",
                "Poses",
                "MotionModule",
            ]
        ] = None,
        sort: Optional[Literal["Highest Rated", "Most Downloaded", "Newest"]] = None,
        period: Optional[Literal["AllTime", "Year", "Month", "Week", "Day"]] = None,
        rating: Optional[int] = None,
        allowNoCredit: Optional[bool] = None,
        allowDerivatives: Optional[bool] = None,
        allowDifferentLicenses: Optional[bool] = None,
        allowCommercialUse: Optional[Literal["None", "Image", "Rent", "Sell"]] = None,
        nsfw: Optional[bool] = None,
    ) -> Iterator[CivitAIModel]:
        """
        Gets models, optionally filtered by some criteria.
        """
        for model in self.get_all(
            "/models",
            limit=limit,
            page=page,
            query=query,
            tag=tag,
            username=username,
            types=types,
            sort=sort,
            period=period,
            rating=rating,
            allowNoCredit=allowNoCredit,
            allowDerivatives=allowDerivatives,
            allowDifferentLicenses=allowDifferentLicenses,
            allowCommercialUse=allowCommercialUse,
            nsfw=nsfw,
        ):
            yield CivitAIModel(**model)
