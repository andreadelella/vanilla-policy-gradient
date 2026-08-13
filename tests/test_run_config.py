import tempfile
import unittest
from pathlib import Path

from vpg.run import _apply_file_config, build_parser


class RunConfigurationTests(unittest.TestCase):
    def test_documented_algorithm_flag_is_accepted(self):
        args = build_parser().parse_args(["--algorithm", "npg"])
        self.assertEqual(args.algorithm, "npg")

    def test_file_defaults_leave_explicit_cli_values_in_control(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory, "config.json")
            config_path.write_text(
                '{"algorithm": "npg", "hidden_sizes": [16, 16]}',
                encoding="utf-8",
            )
            parser = build_parser()
            _apply_file_config(parser, config_path)
            args = parser.parse_args(["--algorithm", "gpomdp"])

        self.assertEqual(args.algorithm, "gpomdp")
        self.assertEqual(args.hidden_sizes, "16,16")


if __name__ == "__main__":
    unittest.main()
