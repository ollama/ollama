import csv
import logging
import pathlib
from optparse import Values
from typing import Iterator, List, NamedTuple, Optional, Tuple

from pip._vendor.packaging.utils import canonicalize_name

from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.metadata import BaseDistribution, get_default_environment
from pip._internal.utils.misc import write_output

logger = logging.getLogger(__name__)


class ShowCommand(Command):
    """
    Show information about one or more installed packages.

    The output is in RFC-compliant mail header format.
    """

    usage = """
      %prog [options] <package> ..."""
    ignore_require_venv = True

    def add_options(self) -> None:
        self.cmd_opts.add_option(
            '-f', '--files',
            dest='files',
            action='store_true',
            default=False,
            help='Show the full list of installed files for each package.')

        self.parser.insert_option_group(0, self.cmd_opts)

    def run(self, options: Values, args: List[str]) -> int:
        if not args:
            logger.warning('ERROR: Please provide a package name or names.')
            return ERROR
        query = args

        results = search_packages_info(query)
        if not print_results(
                results, list_files=options.files, verbose=options.verbose):
            return ERROR
        return SUCCESS


class _PackageInfo(NamedTuple):
    name: str
    version: str
    location: str
    requires: List[str]
    required_by: List[str]
    installer: str
    metadata_version: str
    classifiers: List[str]
    summary: str
    homepage: str
    author: str
    author_email: str
    license: str
    entry_points: List[str]
    files: Optional[List[str]]


def _covert_legacy_entry(entry: Tuple[str, ...], info: Tuple[str, ...]) -> str:
    """Convert a legacy installed-files.txt path into modern RECORD path.

    The legacy format stores paths relative to the info directory, while the
    modern format stores paths relative to the package root, e.g. the
    site-packages directory.

    :param entry: Path parts of the installed-files.txt entry.
    :param info: Path parts of the egg-info directory relative to package root.
    :returns: The converted entry.

    For best compatibility with symlinks, this does not use ``abspath()`` or
    ``Path.resolve()``, but tries to work with path parts:

    1. While ``entry`` starts with ``..``, remove the equal amounts of parts
       from ``info``; if ``info`` is empty, start appending ``..`` instead.
    2. Join the two directly.
    """
    while entry and entry[0] == "..":
        if not info or info[-1] == "..":
            info += ("..",)
        else:
            info = info[:-1]
        entry = entry[1:]
    return str(pathlib.Path(*info, *entry))


def search_packages_info(query: List[str]) -> Iterator[_PackageInfo]:
    """
    Gather details from installed distributions. Print distribution name,
    version, location, and installed files. Installed files requires a
    pip generated 'installed-files.txt' in the distributions '.egg-info'
    directory.
    """
    env = get_default_environment()

    installed = {
        dist.canonical_name: dist
        for dist in env.iter_distributions()
    }
    query_names = [canonicalize_name(name) for name in query]
    missing = sorted(
        [name for name, pkg in zip(query, query_names) if pkg not in installed]
    )
    if missing:
        logger.warning('Package(s) not found: %s', ', '.join(missing))

    def _get_requiring_packages(current_dist: BaseDistribution) -> List[str]:
        return [
            dist.metadata["Name"] or "UNKNOWN"
            for dist in installed.values()
            if current_dist.canonical_name in {
                canonicalize_name(d.name) for d in dist.iter_dependencies()
            }
        ]

    def _files_from_record(dist: BaseDistribution) -> Optional[Iterator[str]]:
        try:
            text = dist.read_text('RECORD')
        except FileNotFoundError:
            return None
        # This extra Path-str cast normalizes entries.
        return (str(pathlib.Path(row[0])) for row in csv.reader(text.splitlines()))

    def _files_from_legacy(dist: BaseDistribution) -> Optional[Iterator[str]]:
        try:
            text = dist.read_text('installed-files.txt')
        except FileNotFoundError:
            return None
        paths = (p for p in text.splitlines(keepends=False) if p)
        root = dist.location
        info = dist.info_directory
        if root is None or info is None:
            return paths
        try:
            info_rel = pathlib.Path(info).relative_to(root)
        except ValueError:  # info is not relative to root.
            return paths
        if not info_rel.parts:  # info *is* root.
            return paths
        return (
            _covert_legacy_entry(pathlib.Path(p).parts, info_rel.parts)
            for p in paths
        )

    for query_name in query_names:
        try:
            dist = installed[query_name]
        except KeyError:
            continue

        try:
            entry_points_text = dist.read_text('entry_points.txt')
            entry_points = entry_points_text.splitlines(keepends=False)
        except FileNotFoundError:
            entry_points = []

        files_iter = _files_from_record(dist) or _files_from_legacy(dist)
        if files_iter is None:
            files: Optional[List[str]] = None
        else:
            files = sorted(files_iter)

        metadata = dist.metadata

        yield _PackageInfo(
            name=dist.raw_name,
            version=str(dist.version),
            location=dist.location or "",
            requires=[req.name for req in dist.iter_dependencies()],
            required_by=_get_requiring_packages(dist),
            installer=dist.installer,
            metadata_version=dist.metadata_version or "",
            classifiers=metadata.get_all("Classifier", []),
            summary=metadata.get("Summary", ""),
            homepage=metadata.get("Home-page", ""),
            author=metadata.get("Author", ""),
            author_email=metadata.get("Author-email", ""),
            license=metadata.get("License", ""),
            entry_points=entry_points,
            files=files,
        )


def print_results(
    distributions: Iterator[_PackageInfo],
    list_files: bool,
    verbose: bool,
) -> bool:
    """
    Print the information from installed distributions found.
    """
    results_printed = False
    for i, dist in enumerate(distributions):
        results_printed = True
        if i > 0:
            write_output("---")

        write_output("Name: %s", dist.name)
        write_output("Version: %s", dist.version)
        write_output("Summary: %s", dist.summary)
        write_output("Home-page: %s", dist.homepage)
        write_output("Author: %s", dist.author)
        write_output("Author-email: %s", dist.author_email)
        write_output("License: %s", dist.license)
        write_output("Location: %s", dist.location)
        write_output("Requires: %s", ', '.join(dist.requires))
        write_output("Required-by: %s", ', '.join(dist.required_by))

        if verbose:
            write_output("Metadata-Version: %s", dist.metadata_version)
            write_output("Installer: %s", dist.installer)
            write_output("Classifiers:")
            for classifier in dist.classifiers:
                write_output("  %s", classifier)
            write_output("Entry-points:")
            for entry in dist.entry_points:
                write_output("  %s", entry.strip())
        if list_files:
            write_output("Files:")
            if dist.files is None:
                write_output("Cannot locate RECORD or installed-files.txt")
            else:
                for line in dist.files:
                    write_output("  %s", line.strip())
    return results_printed
