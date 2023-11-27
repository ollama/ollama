import email.message
import json
import logging
import re
from typing import (
    TYPE_CHECKING,
    Collection,
    Container,
    Iterable,
    Iterator,
    List,
    Optional,
    Union,
)

from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.version import LegacyVersion, Version

from pip._internal.models.direct_url import (
    DIRECT_URL_METADATA_NAME,
    DirectUrl,
    DirectUrlValidationError,
)
from pip._internal.utils.misc import stdlib_pkgs  # TODO: Move definition here.

if TYPE_CHECKING:
    from typing import Protocol

    from pip._vendor.packaging.utils import NormalizedName
else:
    Protocol = object

DistributionVersion = Union[LegacyVersion, Version]

logger = logging.getLogger(__name__)


class BaseEntryPoint(Protocol):
    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def value(self) -> str:
        raise NotImplementedError()

    @property
    def group(self) -> str:
        raise NotImplementedError()


class BaseDistribution(Protocol):
    @property
    def location(self) -> Optional[str]:
        """Where the distribution is loaded from.

        A string value is not necessarily a filesystem path, since distributions
        can be loaded from other sources, e.g. arbitrary zip archives. ``None``
        means the distribution is created in-memory.

        Do not canonicalize this value with e.g. ``pathlib.Path.resolve()``. If
        this is a symbolic link, we want to preserve the relative path between
        it and files in the distribution.
        """
        raise NotImplementedError()

    @property
    def info_directory(self) -> Optional[str]:
        """Location of the .[egg|dist]-info directory.

        Similarly to ``location``, a string value is not necessarily a
        filesystem path. ``None`` means the distribution is created in-memory.

        For a modern .dist-info installation on disk, this should be something
        like ``{location}/{raw_name}-{version}.dist-info``.

        Do not canonicalize this value with e.g. ``pathlib.Path.resolve()``. If
        this is a symbolic link, we want to preserve the relative path between
        it and other files in the distribution.
        """
        raise NotImplementedError()

    @property
    def canonical_name(self) -> "NormalizedName":
        raise NotImplementedError()

    @property
    def version(self) -> DistributionVersion:
        raise NotImplementedError()

    @property
    def direct_url(self) -> Optional[DirectUrl]:
        """Obtain a DirectUrl from this distribution.

        Returns None if the distribution has no `direct_url.json` metadata,
        or if `direct_url.json` is invalid.
        """
        try:
            content = self.read_text(DIRECT_URL_METADATA_NAME)
        except FileNotFoundError:
            return None
        try:
            return DirectUrl.from_json(content)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            DirectUrlValidationError,
        ) as e:
            logger.warning(
                "Error parsing %s for %s: %s",
                DIRECT_URL_METADATA_NAME,
                self.canonical_name,
                e,
            )
            return None

    @property
    def installer(self) -> str:
        raise NotImplementedError()

    @property
    def editable(self) -> bool:
        raise NotImplementedError()

    @property
    def local(self) -> bool:
        raise NotImplementedError()

    @property
    def in_usersite(self) -> bool:
        raise NotImplementedError()

    @property
    def in_site_packages(self) -> bool:
        raise NotImplementedError()

    def read_text(self, name: str) -> str:
        """Read a file in the .dist-info (or .egg-info) directory.

        Should raise ``FileNotFoundError`` if ``name`` does not exist in the
        metadata directory.
        """
        raise NotImplementedError()

    def iter_entry_points(self) -> Iterable[BaseEntryPoint]:
        raise NotImplementedError()

    @property
    def metadata(self) -> email.message.Message:
        """Metadata of distribution parsed from e.g. METADATA or PKG-INFO."""
        raise NotImplementedError()

    @property
    def metadata_version(self) -> Optional[str]:
        """Value of "Metadata-Version:" in distribution metadata, if available."""
        return self.metadata.get("Metadata-Version")

    @property
    def raw_name(self) -> str:
        """Value of "Name:" in distribution metadata."""
        # The metadata should NEVER be missing the Name: key, but if it somehow
        # does not, fall back to the known canonical name.
        return self.metadata.get("Name", self.canonical_name)

    def iter_dependencies(self, extras: Collection[str] = ()) -> Iterable[Requirement]:
        raise NotImplementedError()


class BaseEnvironment:
    """An environment containing distributions to introspect."""

    @classmethod
    def default(cls) -> "BaseEnvironment":
        raise NotImplementedError()

    @classmethod
    def from_paths(cls, paths: Optional[List[str]]) -> "BaseEnvironment":
        raise NotImplementedError()

    def get_distribution(self, name: str) -> Optional["BaseDistribution"]:
        """Given a requirement name, return the installed distributions."""
        raise NotImplementedError()

    def _iter_distributions(self) -> Iterator["BaseDistribution"]:
        """Iterate through installed distributions.

        This function should be implemented by subclass, but never called
        directly. Use the public ``iter_distribution()`` instead, which
        implements additional logic to make sure the distributions are valid.
        """
        raise NotImplementedError()

    def iter_distributions(self) -> Iterator["BaseDistribution"]:
        """Iterate through installed distributions."""
        for dist in self._iter_distributions():
            # Make sure the distribution actually comes from a valid Python
            # packaging distribution. Pip's AdjacentTempDirectory leaves folders
            # e.g. ``~atplotlib.dist-info`` if cleanup was interrupted. The
            # valid project name pattern is taken from PEP 508.
            project_name_valid = re.match(
                r"^([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])$",
                dist.canonical_name,
                flags=re.IGNORECASE,
            )
            if not project_name_valid:
                logger.warning(
                    "Ignoring invalid distribution %s (%s)",
                    dist.canonical_name,
                    dist.location,
                )
                continue
            yield dist

    def iter_installed_distributions(
        self,
        local_only: bool = True,
        skip: Container[str] = stdlib_pkgs,
        include_editables: bool = True,
        editables_only: bool = False,
        user_only: bool = False,
    ) -> Iterator[BaseDistribution]:
        """Return a list of installed distributions.

        :param local_only: If True (default), only return installations
        local to the current virtualenv, if in a virtualenv.
        :param skip: An iterable of canonicalized project names to ignore;
            defaults to ``stdlib_pkgs``.
        :param include_editables: If False, don't report editables.
        :param editables_only: If True, only report editables.
        :param user_only: If True, only report installations in the user
        site directory.
        """
        it = self.iter_distributions()
        if local_only:
            it = (d for d in it if d.local)
        if not include_editables:
            it = (d for d in it if not d.editable)
        if editables_only:
            it = (d for d in it if d.editable)
        if user_only:
            it = (d for d in it if d.in_usersite)
        return (d for d in it if d.canonical_name not in skip)
