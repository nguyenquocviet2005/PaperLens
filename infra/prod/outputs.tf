output "name_servers" {
  description = "Add these as NS records for host 'paperlens' at Mat Bao (ailearn-vaic.cloud)."
  value       = aws_route53_zone.main.name_servers
}

output "ui_url" {
  value = local.ui_origin
}

output "api_url" {
  value = local.api_origin
}

output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "artifacts_bucket" {
  value = aws_s3_bucket.artifacts.bucket
}

output "ecs_cluster" {
  value = aws_ecs_cluster.main.name
}

output "ecs_service" {
  value = aws_ecs_service.api.name
}

output "openai_secret_id" {
  value = aws_secretsmanager_secret.openai_api_key.id
}

output "llm_secret_id" {
  value = aws_secretsmanager_secret.llm_api_key.id
}

output "database_secret_id" {
  value = aws_secretsmanager_secret.database_url.id
}

output "codebuild_project" {
  value = aws_codebuild_project.api.name
}

output "build_source_bucket" {
  value = aws_s3_bucket.build_source.bucket
}

output "compute_mode" {
  value = var.compute_mode
}

output "github_deploy_role_arn" {
  description = "Set as the AWS_DEPLOY_ROLE_ARN repository variable in GitHub"
  value       = aws_iam_role.github_deploy.arn
}

output "dashboard_url" {
  value = "https://${var.region}.console.aws.amazon.com/cloudwatch/home?region=${var.region}#dashboards/dashboard/${aws_cloudwatch_dashboard.main.dashboard_name}"
}

output "langsmith_secret_id" {
  value = aws_secretsmanager_secret.langsmith_api_key.id
}
