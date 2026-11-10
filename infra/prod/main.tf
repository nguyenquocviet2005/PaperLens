terraform {
  required_version = ">= 1.10"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }

  backend "s3" {
    bucket       = "paperlens-tfstate-445137569087"
    key          = "prod/terraform.tfstate"
    region       = "ap-northeast-1"
    encrypt      = true
    use_lockfile = true
  }
}

locals {
  name = "paperlens-prod"
  tags = {
    Project   = "paperlens"
    Env       = "prod"
    ManagedBy = "terraform"
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = local.tags
  }
}

data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  api_domain = "api.${var.domain}"
  ui_origin  = "https://${var.domain}"
  api_origin = "https://${local.api_domain}"

  supabase_auth_url = "https://${var.supabase_project_ref}.supabase.co"

  api_environment = {
    APP_ENV                    = "production"
    LOG_LEVEL                  = "INFO"
    STORAGE_BACKEND            = "s3"
    S3_BUCKET_NAME             = aws_s3_bucket.artifacts.bucket
    AWS_REGION                 = var.region
    DOCLING_OCR_MODE           = "off"
    DOCLING_THREADS            = tostring(local.compute.threads)
    OMP_NUM_THREADS            = tostring(local.compute.threads)
    DOCLING_ACCELERATOR_DEVICE = local.compute.device
    LUNA_ENABLED               = "true"
    LLM_ENABLED                = "true"
    ALLOW_EXTERNAL_API         = "true"
    TEXT_ENRICHMENT_ENABLED    = "true"
    INGEST_AUTO_TEXT_ENRICH    = "true"
    INGEST_ASYNC               = "false"
    LLM_MODEL                  = var.llm_model
    LLM_BASE_URL               = var.llm_base_url
    LUNA_MODEL                 = var.llm_model
    LUNA_BASE_URL              = var.llm_base_url
    EMBEDDING_PROVIDER         = "openai"
    EMBEDDING_MODEL            = var.embedding_model
    EMBEDDING_BASE_URL         = "http://127.0.0.1:${local.embedding_port}/v1"
    EMBEDDING_API_KEY          = "local-tei" # TEI runs without auth; the client requires a value
    EMBEDDING_SEND_DIMENSIONS  = "false"
    EMBEDDING_BATCH_SIZE       = "32"
    EMBEDDING_DIMENSIONS       = "384"
    AGENT_REASONING_ENABLED    = "true"
    AGENT_REASONING_EFFORT     = var.agent_reasoning_effort
    LANGSMITH_ENABLED          = "false"
    LANGSMITH_TRACING          = "false"
    AUTH_ENABLED               = "true"
    SUPABASE_AUTH_URL          = local.supabase_auth_url
    SUPABASE_JWKS_URL          = "${local.supabase_auth_url}/auth/v1/.well-known/jwks.json"
    GUEST_TRIAL_ENABLED        = "true"
    CORS_ALLOWED_ORIGINS       = local.ui_origin
    PAPERLENS_API_BASE         = local.api_origin
  }

  api_secrets = {
    DATABASE_URL = aws_secretsmanager_secret.database_url.arn
    LLM_API_KEY  = aws_secretsmanager_secret.llm_api_key.arn
    LUNA_API_KEY = aws_secretsmanager_secret.llm_api_key.arn
  }
}
