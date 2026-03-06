from click.testing import CliRunner

from kernup.cli.main import cli


def test_cli_help_shows_profile_command() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "profile" in result.output


def test_profile_dry_run_with_gpu_bypass_succeeds() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["profile", "--hf", "Qwen/Qwen2.5-7B", "--dry-run", "--allow-no-gpu"],
    )

    assert result.exit_code == 0
    assert "Bypassing GPU check" in result.output
    assert "Dry-run complete" in result.output


def test_profile_without_dry_run_returns_placeholder_error() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["profile", "--hf", "Qwen/Qwen2.5-7B", "--allow-no-gpu"],
    )

    assert result.exit_code != 0
    assert "Full profiling is not implemented yet" in result.output
