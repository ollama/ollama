import email.message
import logging
import zipfile
from typing import (
    TYPE_CHECKING,
    Collection,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
)

from pip._vendor import pkg_resources
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import parse as parse_version

from pip._internal.utils import misc  # TODO: Move definition here.
from pip._internal.utils.packaging import get_installer, get_metadata
from pip._internal.utils.wheel import pkg_resources_distribution_for_wheel

from .base import BaseDistribution, BaseEntryPoint, BaseEnvironment, DistributionVersion

if TYPE_CHECKING:
    from pip._vendor.packaging.utils import NormalizedName

logger = logging.getLogger(__name__)


class EntryPoint(NamedTuple):
    name: str
    value: str
    group: str


class Distribution(BaseDistribution):
    def __init__(self, dist: pkg_resources.Distribution) -> None:
        self._dist = dist

    @classmethod
    def from_wheel(cls, path: str, name: str) -> "Distribution":
        with zipfile.ZipFile(path, allowZip64=True) as zf:
            dist = pkg_resources_distribution_for_wheel(zf, name, path)
        return cls(dist)

    @property
    def location(self) -> Optional[str]:
        return self._dist.location

    @property
    def info_directory(self) -> Optional[str]:
        return self._dist.egg_info

    @property
    def canonical_name(self) -> "NormalizedName":
        return canonicalize_name(self._dist.project_name)

    @property
    def version(self) -> DistributionVersion:
        return parse_version(self._dist.version)

    @property
    def installer(self) -> str:
        return get_installer(self._dist)

    @property
    def editable(self) -> bool:
        return misc.dist_is_editable(self._dist)

    @property
    def local(self) -> bool:
        return misc.dist_is_local(self._dist)

    @property
    def in_usersite(self) -> bool:
        return misc.dist_in_usersite(self._dist)

    @property
    def in_site_packages(self) -> bool:
        return misc.dist_in_site_packages(self._dist)

    def read_text(self, name: str) -> str:
        if not self._dist.has_metadata(name):
            raise FileNotFoundError(name)
        return self._dist.get_metadata(name)

    def iter_entry_points(self) -> Iterable[BaseEntryPoint]:
        for group, entries in self._dist.get_entry_map().items():
            for name, entry_point in entries.items():
                name, _, value = str(entry_point).partition("=")
                yield EntryPoint(name=name.strip(), value=value.strip(), group=group)

    @property
    def metadata(self) -> email.message.Message:
        return get_metadata(self._dist)

    def iter_dependencies(self, extras: Collection[str] = ()) -> Iterable[Requirement]:
        if extras:  # pkg_resources raises on invalid extras, so we sanitize.
            extras = frozenset(extras).intersection(self._dist.extras)
        return self._dist.requires(extras)


class Environment(BaseEnvironment):
    def __init__(self, ws: pkg_resources.WorkingSet) -> None:
        self._ws = ws

    @classmethod
    def default(cls) -> BaseEnvironment:
        return cls(pkg_resources.working_set)

    @classmethod
    def from_paths(cls, paths: Optional[List[str]]) -> BaseEnvironment:
        return cls(pkg_resources.WorkingSet(paths))

    def _search_distribution(self, name: str) -> Optional[BaseDistribution]:
        """Find a distribution matching the ``name`` in the environment.

        This searches from *all* distributions available in the environment, to
        match the behavior of ``pkg_resources.get_distribution()``.
        """
        canonical_name = canonicalize_name(name)
        for dist in self.iter_distributions():
            if dist.canonical_name == canonical_name:
                return dist
        return None

    def get_distribution(self, name: str) -> Optional[BaseDistribution]:

        # Search the distribution by looking through the working set.
        dist = self._search_distribution(name)
        if dist:
            return dist

        # If distribution could not be found, call working_set.require to
        # update the working set, and try to find the distribution again.
        # This might happen for e.g. when you install a package twice, once
        # using setup.py develop and again using setup.py install. Now when
        # running pip uninstall twice, the package gets removed from the
        # working set in the first uninstall, so we have to populate the
        # working set again so that pip knows about it and the packages gets
        # picked up and is successfully uninstalled the second time too.
        try:
            # We didn't pass in any version specifiers, so this can never
            # raise pkg_resources.VersionConflict.
            self._ws.require(name)
        except pkg_resources.DistributionNotFound:
            return None
        return self._search_distribution(name)

    def _iter_distributions(self) -> Iterator[BaseDistribution]:
        for dist in self._ws:
            yield Distribution(dist)
