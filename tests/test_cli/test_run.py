from click.testing import CliRunner

from noob.cli.run import run as cli_run


def test_cli_run_n(monkeypatch):
    """Run n iterations of a tube, default output to json"""
    runner = CliRunner()
    result = runner.invoke(cli_run, ["testing-basic", "-n", "5", "--output-format", "json"])
    assert result.output.strip() == "[0, 2, 4, 6, 8]"


def test_cli_run_jsonl_output():
    """Output to json lines"""
    runner = CliRunner()
    result = runner.invoke(cli_run, ["testing-basic", "-n", "5", "--output-format", "jsonl"])
    assert result.output.strip() == "0\n2\n4\n6\n8"


def test_cli_run_json_input():
    """Stdin input to run command"""
    runner = CliRunner()
    result = runner.invoke(
        cli_run,
        ["testing-input-process-depends", "--output-format", "json"],
        input='[{"multiply_right": 5}, {"multiply_right": 10}]',
    )
    assert result.output.strip() == "[0, 10]"


def test_cli_run_jsonl_input():
    """Stdin input to run command as jsonl"""
    runner = CliRunner()
    result = runner.invoke(
        cli_run,
        ["testing-input-process-depends", "--output-format", "jsonl"],
        input='[{"multiply_right": 5}, {"multiply_right": 10}]',
    )
    assert result.output.strip() == "0\n10"
