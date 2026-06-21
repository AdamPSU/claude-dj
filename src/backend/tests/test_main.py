import os
import unittest
from unittest.mock import patch

from claude_dj.main import env_flag


class MainCLITests(unittest.TestCase):
    def test_env_flag_accepts_common_truthy_values(self) -> None:
        for value in ["1", "true", "TRUE", "yes", "on"]:
            with self.subTest(value=value), patch.dict(os.environ, {"FLAG": value}):
                self.assertTrue(env_flag("FLAG"))

    def test_env_flag_defaults_false(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(env_flag("FLAG"))


if __name__ == "__main__":
    unittest.main()
