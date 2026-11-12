variable "region" {
  type    = string
  default = "ap-northeast-1"
}

variable "azs" {
  description = "AZs that offer the GPU instance type (g6 is only in 1a/1c in Tokyo)."
  type        = list(string)
  default     = ["ap-northeast-1a", "ap-northeast-1c"]
}

variable "domain" {
  description = "UI hostname; delegated from ailearn-vaic.cloud (Mat Bao) to Route 53."
  type        = string
  default     = "paperlens.ailearn-vaic.cloud"
}

variable "gpu_instance_type" {
  type    = string
  default = "g6.2xlarge"
}

variable "compute_mode" {
  description = "cpu = Fargate (Dockerfile.backend); gpu = EC2 g6 with NVIDIA L4 (Dockerfile.backend.gpu, needs G-instance quota)."
  type        = string
  default     = "cpu"

  validation {
    condition     = contains(["cpu", "gpu"], var.compute_mode)
    error_message = "compute_mode must be \"cpu\" or \"gpu\"."
  }
}

variable "llm_base_url" {
  description = "OpenAI-compatible endpoint for chat completions."
  type        = string
  default     = "https://openrouter.ai/api/v1"
}

variable "llm_model" {
  description = "Must support tool calling and image input (the research agent uses both)."
  type        = string
  default     = "nvidia/nemotron-3.5-super-vl:free"
}

variable "embedding_model" {
  description = "Hugging Face model served by the TEI sidecar; must output 384 dimensions."
  type        = string
  default     = "BAAI/bge-small-en-v1.5"
}

variable "embedding_server_image" {
  description = "Text Embeddings Inference CPU image (pinned)."
  type        = string
  default     = "ghcr.io/huggingface/text-embeddings-inference:cpu-1.9.4"
}

variable "agent_reasoning_effort" {
  type    = string
  default = "low"
}

variable "supabase_project_ref" {
  type    = string
  default = "butwjyfbxuqvakbwismk"
}

variable "alarm_email" {
  description = "Email subscribed to CloudWatch alarm notifications. Empty skips the subscription."
  type        = string
  default     = ""
}

variable "vercel_ipv4" {
  description = "Vercel's recommended A records for the UI domain (Vercel domain config)."
  type        = list(string)
  default     = ["216.198.79.1", "64.29.17.1"]
}

variable "langsmith_enabled" {
  description = "Send agent/LLM traces to LangSmith (needs a value in the langsmith-api-key secret first)"
  type        = bool
  default     = false
}

variable "langsmith_project" {
  description = "LangSmith project that receives production traces"
  type        = string
  default     = "paperlens-prod"
}

variable "github_repository" {
  description = "owner/name of the GitHub repository allowed to deploy through OIDC"
  type        = string
  default     = "nguyenquocviet2005/PaperLens"
}

variable "github_deploy_branch" {
  description = "Only workflows running on this branch may assume the deploy role"
  type        = string
  default     = "master"
}

variable "extra_cors_origins" {
  description = "Other exact browser origins allowed to call the API (e.g. the Vercel project URL)"
  type        = list(string)
  default     = []
}

variable "aux_llm_model" {
  description = "Cheap OpenAI model for history summaries and query rewriting (e.g. gpt-5.4-nano), keyed by the openai-api-key secret. Empty = use llm_model."
  type        = string
  default     = ""
}
