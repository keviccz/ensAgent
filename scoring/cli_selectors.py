from __future__ import annotations

import re


_DOMAIN_TOKEN_RE = re.compile(r"^--domain(\d+)$")
_METHOD_TOKEN_RE = re.compile(r"^--method=([A-Za-z0-9_-]+)$")


def parse_selector_overrides(unknown_args: list[str]) -> tuple[list[int], list[str]]:
    """Parse domain/method selectors from unknown CLI args.

    Supported forms:
      - ``--domain2`` -> domain id 2
      - ``--method=DR-SC`` -> method name (supports letters/digits/_/-)
    """
    selected_domains: set[int] = set()
    selected_methods: set[str] = set()

    for token in unknown_args:
        m_domain = _DOMAIN_TOKEN_RE.match(token)
        if m_domain:
            try:
                selected_domains.add(int(m_domain.group(1)))
            except ValueError:
                continue
            continue

        m_method = _METHOD_TOKEN_RE.match(token)
        if m_method:
            selected_methods.add(m_method.group(1))

    return sorted(selected_domains), sorted(selected_methods)
