from __future__ import annotations

import unittest

from scoring.cli_selectors import parse_selector_overrides


class CliSelectorTests(unittest.TestCase):
    def test_parse_method_supports_hyphen(self) -> None:
        domains, methods = parse_selector_overrides(["--method=DR-SC"])
        self.assertEqual(domains, [])
        self.assertEqual(methods, ["DR-SC"])

    def test_parse_method_supports_underscore(self) -> None:
        domains, methods = parse_selector_overrides(["--method=GraphST_2"])
        self.assertEqual(domains, [])
        self.assertEqual(methods, ["GraphST_2"])

    def test_parse_ignores_invalid_method_tokens(self) -> None:
        domains, methods = parse_selector_overrides(["--method=DR-SC", "--method=bad$name"])
        self.assertEqual(domains, [])
        self.assertEqual(methods, ["DR-SC"])

    def test_parse_domain_tokens(self) -> None:
        domains, methods = parse_selector_overrides(["--domain2", "--domain10", "--domainx"])
        self.assertEqual(domains, [2, 10])
        self.assertEqual(methods, [])


if __name__ == "__main__":
    unittest.main()
