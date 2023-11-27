import collections
import logging
import os
from typing import (
    Container,
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Set,
    Union,
)

from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import Version

from pip._internal.exceptions import BadCommand, InstallationError
from pip._internal.metadata import BaseDistribution, get_environment
from pip._internal.req.constructors import (
    install_req_from_editable,
    install_req_from_line,
)
from pip._internal.req.req_file import COMMENT_RE
from pip._internal.utils.direct_url_helpers import direct_url_as_pep440_direct_reference

logger = logging.getLogger(__name__)


class _EditableInfo(NamedTuple):
    requirement: Optional[str]
    editable: bool
    comments: List[str]


def freeze(
    requirement=None,  # type: Optional[List[str]]
    local_only=False,  # type: bool
    user_only=False,  # type: bool
    paths=None,  # type: Optional[List[str]]
    isolated=False,  # type: bool
    exclude_editable=False,  # type: bool
    skip=()  # type: Container[str]
):
    # type: (...) -> Iterator[str]
    installations = {}  # type: Dict[str, FrozenRequirement]

    dists = get_environment(paths).iter_installed_distributions(
        local_only=local_only,
        skip=(),
        user_only=user_only,
    )
    for dist in dists:
        req = FrozenRequirement.from_dist(dist)
        if exclude_editable and req.editable:
            continue
        installations[req.canonical_name] = req

    if requirement:
        # the options that don't get turned into an InstallRequirement
        # should only be emitted once, even if the same option is in multiple
        # requirements files, so we need to keep track of what has been emitted
        # so that we don't emit it again if it's seen again
        emitted_options = set()  # type: Set[str]
        # keep track of which files a requirement is in so that we can
        # give an accurate warning if a requirement appears multiple times.
        req_files = collections.defaultdict(list)  # type: Dict[str, List[str]]
        for req_file_path in requirement:
            with open(req_file_path) as req_file:
                for line in req_file:
                    if (not line.strip() or
                            line.strip().startswith('#') or
                            line.startswith((
                                '-r', '--requirement',
                                '-f', '--find-links',
                                '-i', '--index-url',
                                '--pre',
                                '--trusted-host',
                                '--process-dependency-links',
                                '--extra-index-url',
                                '--use-feature'))):
                        line = line.rstrip()
                        if line not in emitted_options:
                            emitted_options.add(line)
                            yield line
                        continue

                    if line.startswith('-e') or line.startswith('--editable'):
                        if line.startswith('-e'):
                            line = line[2:].strip()
                        else:
                            line = line[len('--editable'):].strip().lstrip('=')
                        line_req = install_req_from_editable(
                            line,
                            isolated=isolated,
                        )
                    else:
                        line_req = install_req_from_line(
                            COMMENT_RE.sub('', line).strip(),
                            isolated=isolated,
                        )

                    if not line_req.name:
                        logger.info(
                            "Skipping line in requirement file [%s] because "
                            "it's not clear what it would install: %s",
                            req_file_path, line.strip(),
                        )
                        logger.info(
                            "  (add #egg=PackageName to the URL to avoid"
                            " this warning)"
                        )
                    else:
                        line_req_canonical_name = canonicalize_name(
                            line_req.name)
                        if line_req_canonical_name not in installations:
                            # either it's not installed, or it is installed
                            # but has been processed already
                            if not req_files[line_req.name]:
                                logger.warning(
                                    "Requirement file [%s] contains %s, but "
                                    "package %r is not installed",
                                    req_file_path,
                                    COMMENT_RE.sub('', line).strip(),
                                    line_req.name
                                )
                            else:
                                req_files[line_req.name].append(req_file_path)
                        else:
                            yield str(installations[
                                line_req_canonical_name]).rstrip()
                            del installations[line_req_canonical_name]
                            req_files[line_req.name].append(req_file_path)

        # Warn about requirements that were included multiple times (in a
        # single requirements file or in different requirements files).
        for name, files in req_files.items():
            if len(files) > 1:
                logger.warning("Requirement %s included multiple times [%s]",
                               name, ', '.join(sorted(set(files))))

        yield(
            '## The following requirements were added by '
            'pip freeze:'
        )
    for installation in sorted(
            installations.values(), key=lambda x: x.name.lower()):
        if installation.canonical_name not in skip:
            yield str(installation).rstrip()


def _format_as_name_version(dist: BaseDistribution) -> str:
    if isinstance(dist.version, Version):
        return f"{dist.raw_name}=={dist.version}"
    return f"{dist.raw_name}==={dist.version}"


def _get_editable_info(dist: BaseDistribution) -> _EditableInfo:
    """
    Compute and return values (req, editable, comments) for use in
    FrozenRequirement.from_dist().
    """
    if not dist.editable:
        return _EditableInfo(requirement=None, editable=False, comments=[])
    if dist.location is None:
        display = _format_as_name_version(dist)
        logger.warning("Editable requirement not found on disk: %s", display)
        return _EditableInfo(
            requirement=None,
            editable=True,
            comments=[f"# Editable install not found ({display})"],
        )

    location = os.path.normcase(os.path.abspath(dist.location))

    from pip._internal.vcs import RemoteNotFoundError, RemoteNotValidError, vcs

    vcs_backend = vcs.get_backend_for_dir(location)

    if vcs_backend is None:
        display = _format_as_name_version(dist)
        logger.debug(
            'No VCS found for editable requirement "%s" in: %r', display,
            location,
        )
        return _EditableInfo(
            requirement=location,
            editable=True,
            comments=[f'# Editable install with no version control ({display})'],
        )

    vcs_name = type(vcs_backend).__name__

    try:
        req = vcs_backend.get_src_requirement(location, dist.raw_name)
    except RemoteNotFoundError:
        display = _format_as_name_version(dist)
        return _EditableInfo(
            requirement=location,
            editable=True,
            comments=[f'# Editable {vcs_name} install with no remote ({display})'],
        )
    except RemoteNotValidError as ex:
        display = _format_as_name_version(dist)
        return _EditableInfo(
            requirement=location,
            editable=True,
            comments=[
                f"# Editable {vcs_name} install ({display}) with either a deleted "
                f"local remote or invalid URI:",
                f"# '{ex.url}'",
            ],
        )

    except BadCommand:
        logger.warning(
            'cannot determine version of editable source in %s '
            '(%s command not found in path)',
            location,
            vcs_backend.name,
        )
        return _EditableInfo(requirement=None, editable=True, comments=[])

    except InstallationError as exc:
        logger.warning(
            "Error when trying to get requirement for VCS system %s, "
            "falling back to uneditable format", exc
        )
    else:
        return _EditableInfo(requirement=req, editable=True, comments=[])

    logger.warning('Could not determine repository location of %s', location)

    return _EditableInfo(
        requirement=None,
        editable=False,
        comments=['## !! Could not determine repository location'],
    )


class FrozenRequirement:
    def __init__(self, name, req, editable, comments=()):
        # type: (str, Union[str, Requirement], bool, Iterable[str]) -> None
        self.name = name
        self.canonical_name = canonicalize_name(name)
        self.req = req
        self.editable = editable
        self.comments = comments

    @classmethod
    def from_dist(cls, dist: BaseDistribution) -> "FrozenRequirement":
        # TODO `get_requirement_info` is taking care of editable requirements.
        # TODO This should be refactored when we will add detection of
        #      editable that provide .dist-info metadata.
        req, editable, comments = _get_editable_info(dist)
        if req is None and not editable:
            # if PEP 610 metadata is present, attempt to use it
            direct_url = dist.direct_url
            if direct_url:
                req = direct_url_as_pep440_direct_reference(
                    direct_url, dist.raw_name
                )
                comments = []
        if req is None:
            # name==version requirement
            req = _format_as_name_version(dist)

        return cls(dist.raw_name, req, editable, comments=comments)

    def __str__(self):
        # type: () -> str
        req = self.req
        if self.editable:
            req = f'-e {req}'
        return '\n'.join(list(self.comments) + [str(req)]) + '\n'
