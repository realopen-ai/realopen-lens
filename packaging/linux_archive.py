"""Extract our Linux tarball with executable modes and confined relative links."""

from pathlib import Path, PurePosixPath
import tarfile


def extract(archive: Path, destination: Path):
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as packed:
        members, size = [], 0
        for member in packed:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or name.parts[:1] != ("RealOpenLens",):
                raise ValueError("Archive member escapes the Linux application directory")
            if not (member.isfile() or member.isdir() or member.issym()):
                raise ValueError("Unsupported Linux archive member")
            if member.issym():
                link = PurePosixPath(member.linkname)
                if link.is_absolute() or not (
                    destination / name.parent / member.linkname
                ).resolve().is_relative_to(destination / "RealOpenLens"):
                    raise ValueError("Archive link escapes the Linux application directory")
            size += member.size
            members.append(member)
            if len(members) > 50000 or size > 4 * 1024**3:
                raise ValueError("Linux archive exceeds extraction limits")
        packed.extractall(destination, members=members, filter="data")
