# GitHub Actions deploys through OIDC: no long-lived AWS keys in GitHub.
# The role can release a new API image (CodeBuild -> ECR -> new task definition ->
# ECS service) and read what `terraform plan` needs. It cannot change other
# infrastructure: such changes still go through a reviewed local `terraform apply`,
# and the deploy job stops if its plan touches anything beyond the API service.

resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
}

data "aws_iam_policy_document" "github_assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    # Only pushes to the deploy branch: pull requests and other branches cannot deploy.
    # The repository uses GitHub's immutable subject format (owner and repo numeric ids),
    # so a renamed or re-created repository with the same name cannot match.
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["${var.github_oidc_subject_prefix}:ref:refs/heads/${var.github_deploy_branch}"]
    }
  }
}

resource "aws_iam_role" "github_deploy" {
  name                 = "${local.name}-github-deploy"
  assume_role_policy   = data.aws_iam_policy_document.github_assume.json
  max_session_duration = 7200
}

# Terraform refresh reads most resource types in the stack.
resource "aws_iam_role_policy_attachment" "github_read_only" {
  role       = aws_iam_role.github_deploy.name
  policy_arn = "arn:aws:iam::aws:policy/ReadOnlyAccess"
}

locals {
  tf_state_bucket = "paperlens-tfstate-${local.account_id}"
}

data "aws_iam_policy_document" "github_deploy" {
  statement {
    sid       = "TerraformStateList"
    actions   = ["s3:ListBucket"]
    resources = ["arn:aws:s3:::${local.tf_state_bucket}"]
  }
  statement {
    sid     = "TerraformStateReadWrite"
    actions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = [
      "arn:aws:s3:::${local.tf_state_bucket}/prod/terraform.tfstate",
      "arn:aws:s3:::${local.tf_state_bucket}/prod/terraform.tfstate.tflock",
    ]
  }
  statement {
    sid       = "UploadBuildSource"
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.build_source.arn}/source/*"]
  }
  statement {
    sid       = "RunImageBuild"
    actions   = ["codebuild:StartBuild", "codebuild:BatchGetBuilds"]
    resources = [aws_codebuild_project.api.arn]
  }
  statement {
    sid       = "RecordImageTag"
    actions   = ["ssm:PutParameter"]
    resources = [aws_ssm_parameter.api_image_tag.arn]
  }
  statement {
    sid = "RollService"
    actions = [
      "ecs:RegisterTaskDefinition",
      "ecs:DeregisterTaskDefinition",
      "ecs:DeleteTaskDefinitions",
      "ecs:TagResource",
    ]
    resources = ["*"] # task definition ARNs do not exist before registration
  }
  statement {
    sid       = "UpdateService"
    actions   = ["ecs:UpdateService"]
    resources = [aws_ecs_service.api.id]
  }
  statement {
    sid       = "PassTaskRoles"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.api_execution.arn, aws_iam_role.api_task.arn]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["ecs-tasks.amazonaws.com"]
    }
  }
  # ReadOnlyAccess would let the role read uploaded papers and secret values; it needs neither.
  statement {
    sid       = "NoUserData"
    effect    = "Deny"
    actions   = ["s3:GetObject", "s3:GetObjectVersion"]
    resources = ["${aws_s3_bucket.artifacts.arn}/*"]
  }
  statement {
    sid       = "NoSecretValues"
    effect    = "Deny"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "github_deploy" {
  name   = "release-api"
  role   = aws_iam_role.github_deploy.id
  policy = data.aws_iam_policy_document.github_deploy.json
}
