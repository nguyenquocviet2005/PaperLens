"""Build and deploy PaperLens: API to AWS ECS, UI to Vercel.

Run from the repository root:
  python src/ops/deploy_aws.py api    # build image in CodeBuild, push to ECR, roll ECS
  python src/ops/deploy_aws.py ui     # deploy src/frontend to the Vercel "paperlens" project
  python src/ops/deploy_aws.py both

Infrastructure itself is owned by Terraform in infra/prod; this script only
publishes new versions. No secret values are printed. The UI's runtime config
(API URL, Supabase URL and publishable key) lives in the Vercel project's
environment variables and is rendered by src/frontend/vercel-build.sh.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

REGION = "ap-northeast-1"
TF_DIR = Path("infra/prod")
IMAGE_TAG_PARAM = "/paperlens/prod/api-image-tag"
FRONTEND_DIR = Path("src/frontend")


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(">", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def capture(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd[:3])} ...\n{result.stderr.strip()}")
    return result.stdout.strip()


def tf_outputs() -> dict[str, str]:
    raw = capture(["terraform", f"-chdir={TF_DIR}", "output", "-json"])
    return {name: item["value"] for name, item in json.loads(raw).items()}


def image_tag(compute_mode: str) -> str:
    sha = capture(["git", "rev-parse", "--short=12", "HEAD"])
    dirty = capture(["git", "status", "--porcelain"])
    # ECR tags are immutable, so uncommitted builds get a unique suffix. The compute
    # mode is part of the tag because CPU and GPU images come from different Dockerfiles.
    base = f"{sha}-dirty-{int(time.time())}" if dirty else sha
    return f"{base}-{compute_mode}"


def image_exists(repository: str, tag: str) -> bool:
    result = subprocess.run(
        [
            "aws", "ecr", "describe-images", f"--region={REGION}",
            f"--repository-name={repository}", f"--image-ids=imageTag={tag}",
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def package_source(path: Path) -> int:
    """Zip tracked + untracked-but-not-ignored files (so .env never leaves the machine)."""
    names = capture(["git", "ls-files", "-co", "--exclude-standard"]).splitlines()
    count = 0
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            file = Path(name)
            if file.is_file():
                archive.write(file, name)
                count += 1
    return count


def build_image(outputs: dict[str, str], tag: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "source.zip"
        count = package_source(bundle)
        key = f"source/{tag}.zip"
        print(f"Uploading {count} files ({bundle.stat().st_size // 1024} KiB) for build {tag}")
        run(["aws", "s3", "cp", str(bundle), f"s3://{outputs['build_source_bucket']}/{key}"],
            stdout=subprocess.DEVNULL)
    build_id = capture([
        "aws", "codebuild", "start-build", f"--region={REGION}",
        f"--project-name={outputs['codebuild_project']}",
        f"--source-location-override={outputs['build_source_bucket']}/{key}",
        "--environment-variables-override", f"name=IMAGE_TAG,value={tag},type=PLAINTEXT",
        "--query=build.id", "--output=text",
    ])
    print(f"CodeBuild started: {build_id}")
    while True:
        status, phase = capture([
            "aws", "codebuild", "batch-get-builds", f"--region={REGION}", f"--ids={build_id}",
            "--query=builds[0].[buildStatus,currentPhase]", "--output=text",
        ]).split()
        if status != "IN_PROGRESS":
            break
        print(f"  build {phase.lower()}...", flush=True)
        time.sleep(30)
    if status != "SUCCEEDED":
        raise SystemExit(f"CodeBuild {build_id} finished with {status}; see CloudWatch /codebuild logs")
    print(f"Image pushed: {outputs['ecr_repository_url']}:{tag}")


def roll_service(assume_yes: bool) -> None:
    """Plan only the ECS service (the image tag lives in SSM), confirm, then apply that plan."""
    plan_file = "release.tfplan"
    run([
        "terraform", f"-chdir={TF_DIR}", "plan", "-input=false",
        "-target=aws_ecs_service.api", f"-out={plan_file}",
    ])
    try:
        if not assume_yes and input("Apply this plan? [y/N] ").strip().lower() != "y":
            raise SystemExit("Aborted; nothing applied")
        run(["terraform", f"-chdir={TF_DIR}", "apply", "-input=false", plan_file])
    finally:
        (TF_DIR / plan_file).unlink(missing_ok=True)


def deploy_api(tag: str | None, assume_yes: bool) -> None:
    outputs = tf_outputs()
    repository = outputs["ecr_repository_url"].split("/", 1)[1]
    tag = tag or image_tag(outputs["compute_mode"])

    if image_exists(repository, tag):
        print(f"Image tag {tag} already in ECR; skipping build")
    else:
        build_image(outputs, tag)

    run([
        "aws", "ssm", "put-parameter", f"--region={REGION}",
        f"--name={IMAGE_TAG_PARAM}", f"--value={tag}", "--overwrite",
    ], stdout=subprocess.DEVNULL)
    roll_service(assume_yes)
    print("Waiting for the ECS service to become stable (first start can take ~10 min)...")
    run([
        "aws", "ecs", "wait", "services-stable", f"--region={REGION}",
        f"--cluster={outputs['ecs_cluster']}", f"--services={outputs['ecs_service']}",
    ])
    print(f"API deployed: {outputs['api_url']} ({tag})")


def deploy_ui() -> None:
    if not (FRONTEND_DIR / ".vercel" / "project.json").exists():
        raise SystemExit(
            "src/frontend is not linked to Vercel; run `vercel link --project paperlens` there first"
        )
    run(["vercel", "deploy", "--prod", "--yes"], cwd=FRONTEND_DIR, stdout=subprocess.DEVNULL)
    print("UI deployed: https://paperlens.ailearn-vaic.cloud")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["api", "ui", "both"])
    parser.add_argument("--tag", help="Deploy an existing ECR image tag instead of building")
    parser.add_argument("--build-only", action="store_true", help="Build and push the API image only")
    parser.add_argument("--yes", action="store_true", help="Apply the release plan without prompting")
    args = parser.parse_args()
    if args.build_only:
        outputs = tf_outputs()
        tag = args.tag or image_tag(outputs["compute_mode"])
        build_image(outputs, tag)
        print(f"TAG={tag}")
        return 0
    if args.target in {"api", "both"}:
        deploy_api(args.tag, args.yes)
    if args.target in {"ui", "both"}:
        deploy_ui()
    return 0


if __name__ == "__main__":
    sys.exit(main())
