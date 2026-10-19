"""Dummy canary rollout planner for the k8s-rollout skill."""

import argparse


def main() -> int:
    """Print the canary plan for a deployment."""
    parser = argparse.ArgumentParser(prog="rollout")
    parser.add_argument("action", choices=["plan", "watch"])
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--image", default="")
    parser.add_argument("--namespace", default="default")
    args = parser.parse_args()
    for pct in (10, 50, 100):
        print(f"{args.action}: {args.namespace}/{args.deployment} -> {pct}% {args.image}".rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
