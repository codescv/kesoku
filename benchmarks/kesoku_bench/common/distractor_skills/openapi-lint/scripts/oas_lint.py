"""Minimal OpenAPI linter (JSON only) for the openapi-lint skill."""

import argparse
import json


def main() -> int:
    """Report operations without an operationId."""
    parser = argparse.ArgumentParser(prog="oas_lint")
    parser.add_argument("spec")
    parser.add_argument("--fail-on", default="error", choices=["error", "warning"])
    parser.add_argument("--rules", default="all")
    args = parser.parse_args()
    with open(args.spec, encoding="utf-8") as fh:
        doc = json.load(fh)
    findings = 0
    for path, ops in doc.get("paths", {}).items():
        for method, op in ops.items():
            if "operationId" not in op:
                findings += 1
                print(f"{path} {method.upper()} missing-operation-id")
    return 1 if findings and args.fail_on == "warning" else 0


if __name__ == "__main__":
    raise SystemExit(main())
