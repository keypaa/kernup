"""Main CLI entrypoint for Kernup."""

from __future__ import annotations

import click

from kernup import __version__
from kernup.cli.optimize import optimize_command
from kernup.cli.profile import profile_command


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="kernup")
def cli() -> None:
    """Kernup command-line interface."""


cli.add_command(profile_command)
cli.add_command(optimize_command)


if __name__ == "__main__":
    cli()
