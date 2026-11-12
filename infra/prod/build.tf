# Container builds run in CodeBuild (the GPU image is ~10 GB, too large for a laptop).
# src/ops/deploy_aws.py uploads a source zip and starts a build with IMAGE_TAG set.

resource "aws_s3_bucket" "build_source" {
  bucket        = "${local.name}-build-src-${local.account_id}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "build_source" {
  bucket                  = aws_s3_bucket.build_source.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "build_source" {
  bucket = aws_s3_bucket.build_source.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "build_source" {
  bucket = aws_s3_bucket.build_source.id
  rule {
    id     = "expire-sources"
    status = "Enabled"
    filter {}
    expiration {
      days = 14
    }
  }
}

resource "aws_cloudwatch_log_group" "build" {
  name              = "/codebuild/${local.name}-api"
  retention_in_days = 30
}

data "aws_iam_policy_document" "codebuild_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["codebuild.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
  }
}

resource "aws_iam_role" "codebuild" {
  name               = "${local.name}-codebuild"
  assume_role_policy = data.aws_iam_policy_document.codebuild_assume.json
}

data "aws_iam_policy_document" "codebuild" {
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.build.arn}:*"]
  }
  statement {
    sid       = "ReadSource"
    actions   = ["s3:GetObject", "s3:GetObjectVersion"]
    resources = ["${aws_s3_bucket.build_source.arn}/*"]
  }
  statement {
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    sid = "EcrPush"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:CompleteLayerUpload",
      "ecr:GetDownloadUrlForLayer",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
    ]
    resources = [aws_ecr_repository.api.arn]
  }
}

resource "aws_iam_role_policy" "codebuild" {
  name   = "build-and-push-api"
  role   = aws_iam_role.codebuild.id
  policy = data.aws_iam_policy_document.codebuild.json
}

resource "aws_codebuild_project" "api" {
  name          = "${local.name}-api"
  description   = "Builds the PaperLens API image and pushes it to ECR"
  service_role  = aws_iam_role.codebuild.arn
  build_timeout = 120

  source {
    type     = "S3"
    location = "${aws_s3_bucket.build_source.bucket}/source/latest.zip"
    buildspec = yamlencode({
      version = "0.2"
      phases = {
        pre_build = {
          commands = [
            "aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $${REPOSITORY_URL%%/*}",
          ]
        }
        build = {
          commands = [
            "docker build --platform=linux/amd64 -f $DOCKERFILE -t $REPOSITORY_URL:$IMAGE_TAG .",
          ]
        }
        post_build = {
          commands = [
            "test \"$CODEBUILD_BUILD_SUCCEEDING\" = 1",
            "docker push $REPOSITORY_URL:$IMAGE_TAG",
          ]
        }
      }
    })
  }

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    compute_type    = "BUILD_GENERAL1_LARGE"
    image           = "aws/codebuild/amazonlinux-x86_64-standard:5.0"
    type            = "LINUX_CONTAINER"
    privileged_mode = true

    environment_variable {
      name  = "REPOSITORY_URL"
      value = aws_ecr_repository.api.repository_url
    }

    environment_variable {
      name  = "DOCKERFILE"
      value = local.compute.dockerfile
    }

    environment_variable {
      name  = "IMAGE_TAG"
      value = "override-at-start"
    }
  }

  logs_config {
    cloudwatch_logs {
      group_name = aws_cloudwatch_log_group.build.name
    }
  }
}
