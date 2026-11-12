# Committed on purpose: CI deploys and local applies must read the same values, or a
# release would silently revert them. Personal values (alarm_email) stay in the
# gitignored prod.auto.tfvars locally and in the TF_VAR_ALARM_EMAIL GitHub secret.
compute_mode      = "cpu"
langsmith_enabled = true
# The Vercel project URL also serves the production UI.
extra_cors_origins = ["https://paperlens-six.vercel.app"]
# Cheap OpenAI model for conversation summaries and query rewriting; empty uses llm_model
# (OpenRouter). Needs a funded key in paperlens/prod/openai-api-key.
aux_llm_model = ""
