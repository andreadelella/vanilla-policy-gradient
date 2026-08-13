import unittest

from vpg.config import parse_hidden_sizes


class HiddenSizeParsingTests(unittest.TestCase):
    def test_parses_comma_separated_sizes(self):
        self.assertEqual(parse_hidden_sizes("32, 16"), [32, 16])

    def test_rejects_empty_non_numeric_and_non_positive_sizes(self):
        for value in ("", "8,0", "-1,8", "eight,8"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_hidden_sizes(value)


if __name__ == "__main__":
    unittest.main()
