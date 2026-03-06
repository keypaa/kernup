"""Main CLI entrypoint for Kernup."""

from __future__ import annotations

import click

from kernup import __version__
from kernup.cli.bench import bench_command
from kernup.cli.optimize import optimize_command
from kernup.cli.patch import patch_command
from kernup.cli.profile import profile_command
from kernup.cli.status import clean_command, status_command


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="kernup")
def cli() -> None:
    """Kernup command-line interface."""


cli.add_command(profile_command)
cli.add_command(optimize_command)
cli.add_command(patch_command)
cli.add_command(bench_command)
cli.add_command(status_command)
cli.add_command(clean_command)


if __name__ == "__main__":
    cli()
