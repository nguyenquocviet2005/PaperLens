"""The release plan guard in src/ops/deploy_aws.py (CD must not change other infrastructure)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "ops" / "deploy_aws.py"
_spec = importlib.util.spec_from_file_location("deploy_aws", _PATH)
deploy_aws = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(deploy_aws)


def _plan(*changes):
    return {"resource_changes": [{"address": a, "change": {"actions": list(acts)}} for a, acts in changes]}


def test_normal_release_is_allowed():
    plan = _plan(
        ("aws_ecs_task_definition.api", ["delete", "create"]),
        ("aws_ecs_service.api", ["update"]),
        ("aws_ssm_parameter.api_image_tag", ["no-op"]),
        ("data.aws_caller_identity.current", ["read"]),
    )
    assert deploy_aws.unexpected_changes(plan) == []


def test_other_infrastructure_changes_are_refused():
    plan = _plan(
        ("aws_ecs_service.api", ["update"]),
        ("aws_secretsmanager_secret.langsmith_api_key", ["create"]),
        ("aws_lb.api", ["delete", "create"]),
    )
    assert deploy_aws.unexpected_changes(plan) == [
        "aws_secretsmanager_secret.langsmith_api_key: create",
        "aws_lb.api: create/delete",
    ]


def test_replacing_the_service_is_refused():
    plan = _plan(("aws_ecs_service.api", ["delete", "create"]))
    assert deploy_aws.unexpected_changes(plan) == ["aws_ecs_service.api: create/delete"]
