"""Tests for config environment variable parsing."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config


class TestEnvBool(unittest.TestCase):
    def test_returns_default_when_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(config._env_bool("MISSING_FLAG", True))
            self.assertFalse(config._env_bool("MISSING_FLAG", False))

    def test_truthy_values(self) -> None:
        truthy = ("true", "TRUE", "True", "1", "yes", "YES", "on", "ON", "  true  ")
        for value in truthy:
            with self.subTest(value=value):
                with patch.dict(os.environ, {"TEST_BOOL": value}, clear=True):
                    self.assertTrue(config._env_bool("TEST_BOOL", False))

    def test_falsy_values(self) -> None:
        falsy = ("false", "0", "no", "off", "", "maybe", "true-ish")
        for value in falsy:
            with self.subTest(value=value):
                with patch.dict(os.environ, {"TEST_BOOL": value}, clear=True):
                    self.assertFalse(config._env_bool("TEST_BOOL", True))


class TestEnvPath(unittest.TestCase):
    def test_returns_default_when_unset(self) -> None:
        default = Path("/default/path")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(config._env_path("MISSING_PATH", default), default)

    def test_parses_path_from_env(self) -> None:
        with patch.dict(os.environ, {"TEST_PATH": "Data/custom"}, clear=True):
            self.assertEqual(config._env_path("TEST_PATH", Path("ignored")), Path("Data/custom"))


class TestLoadEnvFile(unittest.TestCase):
    def _write_env(self, contents: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False)
        tmp.write(contents)
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        return Path(tmp.name)

    def test_parses_basic_assignments(self) -> None:
        env_file = self._write_env(
            "\n".join(
                [
                    "# leading comment",
                    "FOO=bar",
                    "ITS_COMING_HOME=true # inline comment",
                    'QUOTED="hello world"',
                    "SPACED = value ",
                ]
            )
        )
        with patch.dict(os.environ, {}, clear=True):
            config._load_env_file(env_file)
            self.assertEqual(os.environ["FOO"], "bar")
            self.assertEqual(os.environ["ITS_COMING_HOME"], "true")
            self.assertEqual(os.environ["QUOTED"], "hello world")
            self.assertEqual(os.environ["SPACED"], "value")

    def test_skips_blank_and_comment_lines(self) -> None:
        env_file = self._write_env("# only comment\n\n# another\n")
        with patch.dict(os.environ, {}, clear=True):
            config._load_env_file(env_file)
            self.assertEqual(os.environ, {})

    def test_skips_lines_without_equals(self) -> None:
        env_file = self._write_env("NOT_A_VAR\nVALID=yes\n")
        with patch.dict(os.environ, {}, clear=True):
            config._load_env_file(env_file)
            self.assertNotIn("NOT_A_VAR", os.environ)
            self.assertEqual(os.environ["VALID"], "yes")

    def test_does_not_override_existing_env(self) -> None:
        env_file = self._write_env("FOO=from_file\n")
        with patch.dict(os.environ, {"FOO": "from_shell"}, clear=True):
            config._load_env_file(env_file)
            self.assertEqual(os.environ["FOO"], "from_file")


class TestProjectEnvFile(unittest.TestCase):
    """Smoke test that the repo's real .env.example values parse as expected."""

    def test_env_example_parses_its_coming_home_when_set(self) -> None:
        env_file = ROOT / ".env.example"
        if not env_file.is_file():
            self.skipTest(".env.example not found")

        with patch.dict(os.environ, {}, clear=True):
            config._load_env_file(env_file)
            self.assertNotIn("ITS_COMING_HOME", os.environ)

        with patch.dict(os.environ, {}, clear=True):
            config._load_env_file(
                self._env_with_active_line(env_file, "ITS_COMING_HOME", "true")
            )
            self.assertTrue(config._env_bool("ITS_COMING_HOME", False))

    def _env_with_active_line(
        self, example_path: Path, key: str, value: str
    ) -> Path:
        lines = example_path.read_text(encoding="utf-8").splitlines()
        active = [
            f"{key}={value}" if line.strip().startswith(f"# {key}=") else line
            for line in lines
        ]
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False)
        tmp.write("\n".join(active))
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        return Path(tmp.name)


class TestComingHomeFlag(unittest.TestCase):
    def test_project_env_overrides_preset_false(self) -> None:
        with patch.dict(os.environ, {"ITS_COMING_HOME": "false"}, clear=True):
            if "config" in sys.modules:
                del sys.modules["config"]
            sys.path.insert(0, str(ROOT))
            import config as reloaded_config

            self.assertTrue(reloaded_config.ITS_COMING_HOME)
            del sys.modules["config"]
            sys.modules["config"] = config

    def test_explicit_override_beats_module_default(self) -> None:
        with patch.object(config, "ITS_COMING_HOME", False):
            self.assertEqual(
                config.adjust_coming_home_group_goals(
                    "Senegal", "England", 2, 1, its_coming_home=True
                ),
                (2, 2),
            )

            rng = np.random.default_rng(0)
            winner, loser, _, _, decided_by = config.resolve_coming_home_knockout(
                "Senegal", "England", 2.0, 1.0, rng, its_coming_home=True
            )
            self.assertEqual((winner, loser, decided_by), ("England", "Senegal", "Penalties"))

    def test_module_reassignment_is_respected(self) -> None:
        original = config.ITS_COMING_HOME
        try:
            config.ITS_COMING_HOME = True
            self.assertEqual(
                config.adjust_coming_home_group_goals("France", "England", 2, 1),
                (2, 2),
            )
        finally:
            config.ITS_COMING_HOME = original


if __name__ == "__main__":
    unittest.main()
