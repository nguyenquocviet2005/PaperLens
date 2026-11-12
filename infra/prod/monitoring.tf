resource "aws_sns_topic" "alarms" {
  name = "${local.name}-alarms"
}

resource "aws_sns_topic_subscription" "alarm_email" {
  count     = var.alarm_email == "" ? 0 : 1
  topic_arn = aws_sns_topic.alarms.arn
  protocol  = "email"
  endpoint  = var.alarm_email
}

locals {
  alb_dimensions = {
    LoadBalancer = aws_lb.api.arn_suffix
  }
  target_dimensions = {
    LoadBalancer = aws_lb.api.arn_suffix
    TargetGroup  = aws_lb_target_group.api.arn_suffix
  }
  service_dimensions = {
    ClusterName = aws_ecs_cluster.main.name
    ServiceName = aws_ecs_service.api.name
  }
}

resource "aws_cloudwatch_metric_alarm" "api_5xx" {
  alarm_name          = "${local.name}-api-5xx"
  alarm_description   = "API returned more than 10 5xx responses in 5 minutes"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "HTTPCode_Target_5XX_Count"
  dimensions          = local.target_dimensions
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 10
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
  ok_actions          = [aws_sns_topic.alarms.arn]
}

resource "aws_cloudwatch_metric_alarm" "api_unhealthy" {
  alarm_name          = "${local.name}-api-unhealthy-targets"
  alarm_description   = "API target failing ALB health checks"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "UnHealthyHostCount"
  dimensions          = local.target_dimensions
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 5
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
  ok_actions          = [aws_sns_topic.alarms.arn]
}

resource "aws_cloudwatch_metric_alarm" "alb_5xx" {
  alarm_name          = "${local.name}-alb-5xx"
  alarm_description   = "Load balancer itself returned 5xx (no healthy targets, timeouts)"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "HTTPCode_ELB_5XX_Count"
  dimensions          = local.alb_dimensions
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 10
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
}

resource "aws_cloudwatch_metric_alarm" "api_cpu" {
  alarm_name          = "${local.name}-api-cpu-high"
  namespace           = "AWS/ECS"
  metric_name         = "CPUUtilization"
  dimensions          = local.service_dimensions
  statistic           = "Average"
  period              = 300
  evaluation_periods  = 3
  threshold           = 90
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
}

resource "aws_cloudwatch_metric_alarm" "api_memory" {
  alarm_name          = "${local.name}-api-memory-high"
  namespace           = "AWS/ECS"
  metric_name         = "MemoryUtilization"
  dimensions          = local.service_dimensions
  statistic           = "Average"
  period              = 300
  evaluation_periods  = 3
  threshold           = 90
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
}

# ---------------------------------------------------------------------------
# Application metrics (CloudWatch Embedded Metric Format, namespace "PaperLens")
# emitted by app/infrastructure/observability.py.
# ---------------------------------------------------------------------------

locals {
  app_ns      = "PaperLens"
  agent_route = "POST /agent/stream"
  log_group   = aws_cloudwatch_log_group.api.name
}

resource "aws_cloudwatch_metric_alarm" "llm_errors" {
  alarm_name          = "${local.name}-llm-errors"
  alarm_description   = "5+ failed LLM/VLM calls in 10 minutes (provider outage, quota, bad key)"
  namespace           = local.app_ns
  metric_name         = "LLMErrors"
  statistic           = "Sum"
  period              = 600
  evaluation_periods  = 1
  threshold           = 5
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
  ok_actions          = [aws_sns_topic.alarms.arn]
}

resource "aws_cloudwatch_metric_alarm" "ingest_failures" {
  alarm_name          = "${local.name}-ingest-failures"
  alarm_description   = "2+ paper uploads failed or were rejected as incomplete in 30 minutes"
  namespace           = local.app_ns
  metric_name         = "IngestFailures"
  statistic           = "Sum"
  period              = 1800
  evaluation_periods  = 1
  threshold           = 2
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
}

resource "aws_cloudwatch_metric_alarm" "agent_first_byte" {
  alarm_name          = "${local.name}-agent-slow-start"
  alarm_description   = "p95 time until the chat stream starts is above 20 s for 15 minutes"
  namespace           = local.app_ns
  metric_name         = "TimeToFirstByte"
  dimensions          = { Route = local.agent_route }
  extended_statistic  = "p95"
  period              = 300
  evaluation_periods  = 3
  threshold           = 20000
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
}

resource "aws_cloudwatch_log_metric_filter" "api_error_logs" {
  name           = "${local.name}-api-error-logs"
  log_group_name = local.log_group
  pattern        = "{ $.level = \"ERROR\" }"
  metric_transformation {
    name          = "ErrorLogLines"
    namespace     = local.app_ns
    value         = "1"
    default_value = "0"
  }
}

resource "aws_cloudwatch_metric_alarm" "api_error_logs" {
  alarm_name          = "${local.name}-api-error-logs"
  alarm_description   = "More than 10 ERROR log lines in 5 minutes"
  namespace           = local.app_ns
  metric_name         = "ErrorLogLines"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 10
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
}

# Saved Logs Insights queries (CloudWatch > Logs Insights > Saved queries).
resource "aws_cloudwatch_query_definition" "errors" {
  name            = "paperlens/errors"
  log_group_names = [local.log_group]
  query_string    = <<-QUERY
    fields @timestamp, request_id, logger, message, exception
    | filter level = "ERROR"
    | sort @timestamp desc
    | limit 100
  QUERY
}

resource "aws_cloudwatch_query_definition" "request_trace" {
  name            = "paperlens/one-request (replace the id)"
  log_group_names = [local.log_group]
  query_string    = <<-QUERY
    fields @timestamp, level, logger, message, route, status, duration_ms
    | filter request_id = "PASTE-X-REQUEST-ID-HERE"
    | sort @timestamp asc
  QUERY
}

resource "aws_cloudwatch_query_definition" "slow_routes" {
  name            = "paperlens/latency-by-route"
  log_group_names = [local.log_group]
  query_string    = <<-QUERY
    filter logger = "paperlens.access"
    | stats count() as requests, pct(duration_ms, 50) as p50_ms, pct(duration_ms, 95) as p95_ms,
            max(duration_ms) as max_ms by route
    | sort p95_ms desc
  QUERY
}

resource "aws_cloudwatch_query_definition" "llm_usage" {
  name            = "paperlens/llm-usage-by-model"
  log_group_names = [local.log_group]
  query_string    = <<-QUERY
    filter ispresent(LLMCalls) and ispresent(Model)
    | stats sum(LLMCalls) as calls, sum(LLMErrors) as errors, sum(LLMInputTokens) as input_tokens,
            sum(LLMOutputTokens) as output_tokens, pct(LLMLatency, 95) as p95_ms by Model
  QUERY
}

resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = local.name
  dashboard_body = jsonencode({
    start          = "-PT24H"
    periodOverride = "inherit"
    widgets = [
      {
        type = "text", x = 0, y = 0, width = 24, height = 2
        properties = {
          markdown = "## PaperLens production\nTraffic and errors from the load balancer; everything below the first row comes from the app (namespace `PaperLens`). Find one request: Logs Insights saved query `paperlens/one-request` with the `X-Request-ID` response header. LLM traces: LangSmith project `${var.langsmith_project}`."
        }
      },
      {
        type = "metric", x = 0, y = 2, width = 8, height = 6
        properties = {
          title = "Requests and errors (ALB)", region = var.region, stat = "Sum", period = 300
          metrics = [
            ["AWS/ApplicationELB", "RequestCount", "LoadBalancer", aws_lb.api.arn_suffix, { label = "requests" }],
            [".", "HTTPCode_Target_5XX_Count", "TargetGroup", aws_lb_target_group.api.arn_suffix, "LoadBalancer", aws_lb.api.arn_suffix, { label = "5xx (app)", yAxis = "right" }],
            [".", "HTTPCode_ELB_5XX_Count", "LoadBalancer", aws_lb.api.arn_suffix, { label = "5xx (ALB)", yAxis = "right" }],
          ]
        }
      },
      {
        type = "metric", x = 8, y = 2, width = 8, height = 6
        properties = {
          title   = "p95 latency by route (ms)", region = var.region, period = 300
          metrics = [[{ expression = "SEARCH('{PaperLens,Route} MetricName=\"RequestLatency\"', 'p95', 300)", id = "lat", label = "" }]]
        }
      },
      {
        type = "metric", x = 16, y = 2, width = 8, height = 6
        properties = {
          title = "Chat: time until the stream starts (ms)", region = var.region, period = 300
          metrics = [
            [local.app_ns, "TimeToFirstByte", "Route", local.agent_route, { stat = "p50", label = "p50" }],
            ["...", { stat = "p95", label = "p95" }],
          ]
        }
      },
      {
        type = "metric", x = 0, y = 8, width = 8, height = 6
        properties = {
          title = "LLM tokens by model", region = var.region, period = 3600, stacked = true, view = "timeSeries"
          metrics = [
            [{ expression = "SEARCH('{PaperLens,Model} MetricName=\"LLMInputTokens\"', 'Sum', 3600)", id = "inp", label = "in" }],
            [{ expression = "SEARCH('{PaperLens,Model} MetricName=\"LLMOutputTokens\"', 'Sum', 3600)", id = "out", label = "out" }],
          ]
        }
      },
      {
        type = "metric", x = 8, y = 8, width = 8, height = 6
        properties = {
          title = "LLM calls and errors", region = var.region, stat = "Sum", period = 300
          metrics = [
            [local.app_ns, "LLMCalls", { label = "calls" }],
            [".", "LLMErrors", { label = "errors", yAxis = "right" }],
          ]
        }
      },
      {
        type = "metric", x = 16, y = 8, width = 8, height = 6
        properties = {
          title   = "LLM p95 latency by model (ms)", region = var.region, period = 300
          metrics = [[{ expression = "SEARCH('{PaperLens,Model} MetricName=\"LLMLatency\"', 'p95', 300)", id = "llm", label = "" }]]
        }
      },
      {
        type = "metric", x = 0, y = 14, width = 8, height = 6
        properties = {
          title = "Paper ingestion", region = var.region, period = 3600
          metrics = [
            [local.app_ns, "Ingests", { stat = "Sum", label = "uploads" }],
            [".", "IngestFailures", { stat = "Sum", label = "failed/incomplete" }],
            [".", "IngestDuration", { stat = "p95", label = "p95 seconds", yAxis = "right" }],
          ]
        }
      },
      {
        type = "metric", x = 8, y = 14, width = 8, height = 6
        properties = {
          title   = "Answers by confidence", region = var.region, period = 3600, stacked = true
          metrics = [[{ expression = "SEARCH('{PaperLens,Confidence,Mode} MetricName=\"QAAnswers\"', 'Sum', 3600)", id = "qa", label = "" }]]
        }
      },
      {
        type = "metric", x = 16, y = 14, width = 8, height = 6
        properties = {
          title = "Retrieval", region = var.region, period = 300
          metrics = [
            [local.app_ns, "RetrievalLatency", { stat = "p95", label = "p95 ms" }],
            [".", "EvidenceTokens", { stat = "Average", label = "avg evidence tokens", yAxis = "right" }],
          ]
        }
      },
      {
        type = "metric", x = 0, y = 20, width = 8, height = 6
        properties = {
          title   = "Guardrail blocks by code", region = var.region, period = 3600, stacked = true
          metrics = [[{ expression = "SEARCH('{PaperLens,Code} MetricName=\"GuardrailBlocks\"', 'Sum', 3600)", id = "gr", label = "" }]]
        }
      },
      {
        type = "metric", x = 8, y = 20, width = 8, height = 6
        properties = {
          title = "API container CPU / memory (%)", region = var.region, stat = "Average", period = 300
          metrics = [
            ["AWS/ECS", "CPUUtilization", "ClusterName", aws_ecs_cluster.main.name, "ServiceName", aws_ecs_service.api.name],
            [".", "MemoryUtilization", ".", ".", ".", "."],
          ]
        }
      },
      {
        type = "metric", x = 0, y = 26, width = 12, height = 6
        properties = {
          title = "Agent context per turn (tokens)", region = var.region, stat = "Average", period = 3600
          metrics = [
            [local.app_ns, "ContextHistoryTokens", { label = "recent turns" }],
            [".", "ContextMemoryTokens", { label = "memory (state + summary)" }],
            [".", "HistoryTurnsDropped", { label = "turns outside the window", yAxis = "right" }],
          ]
        }
      },
      {
        type = "metric", x = 12, y = 26, width = 12, height = 6
        properties = {
          title   = "History summaries", region = var.region, period = 3600, stat = "Sum"
          metrics = [[{ expression = "SEARCH('{PaperLens,Method} MetricName=\"HistorySummaries\"', 'Sum', 3600)", id = "sum", label = "" }]]
        }
      },
      {
        type = "log", x = 16, y = 20, width = 8, height = 6
        properties = {
          title = "Latest errors", region = var.region, view = "table"
          query = "SOURCE '${local.log_group}' | fields @timestamp, request_id, logger, message | filter level = \"ERROR\" | sort @timestamp desc | limit 20"
        }
      },
    ]
  })
}
