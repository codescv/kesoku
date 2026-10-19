---
name: k8s-rollout
description: "Plan and watch a Kubernetes deployment rollout with canary steps and automatic rollback."
version: "2.0.1"
---

# k8s-rollout

Wraps `kubectl rollout` with a canary plan: 10% -> 50% -> 100%, pausing between steps and rolling back
when the error-rate SLO is breached.

## Usage

```bash
uv run python $SKILL_DIR/scripts/rollout.py plan --deployment api --image api:1.42
uv run python $SKILL_DIR/scripts/rollout.py watch --deployment api --namespace prod
```

Requires a configured `kubectl` context. Always run `plan` first and paste it into the change ticket.
