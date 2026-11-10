resource "aws_ecs_cluster" "main" {
  name = local.name

  setting {
    name  = "containerInsights"
    value = "enhanced"
  }
}

# Currently deployed API image tag. Written by src/ops/deploy_aws.py, so any
# `terraform apply` keeps the running version instead of reverting it.
resource "aws_ssm_parameter" "api_image_tag" {
  name           = "/paperlens/prod/api-image-tag"
  type           = "String"
  insecure_value = "unset"

  lifecycle {
    ignore_changes = [insecure_value]
  }
}

data "aws_ssm_parameter" "ecs_gpu_ami" {
  name = "/aws/service/ecs/optimized-ami/amazon-linux-2023/gpu/recommended/image_id"
}

locals {
  api_image_tag  = aws_ssm_parameter.api_image_tag.insecure_value
  embedding_port = 8081
  use_gpu        = var.compute_mode == "gpu"

  compute = local.use_gpu ? {
    capacity_provider = aws_ecs_capacity_provider.gpu.name
    compatibility     = "EC2"
    cpu               = 7168
    memory            = 28672
    threads           = 4
    device            = "cuda"
    dockerfile        = "Dockerfile.backend.gpu"
    } : {
    capacity_provider = "FARGATE"
    compatibility     = "FARGATE"
    cpu               = 8192
    memory            = 32768
    threads           = 8
    device            = "cpu"
    dockerfile        = "Dockerfile.backend"
  }

  # Host directory persisted across task restarts so Docling/HF models download once per instance.
  model_cache_host_path = "/var/lib/paperlens/cache"

  ecs_user_data = <<-EOT
    #!/bin/bash
    set -euo pipefail
    cat >> /etc/ecs/ecs.config <<EOF
    ECS_CLUSTER=${aws_ecs_cluster.main.name}
    ECS_ENABLE_GPU_SUPPORT=true
    ECS_AWSVPC_BLOCK_IMDS=true
    ECS_ENABLE_CONTAINER_METADATA=true
    ECS_IMAGE_PULL_BEHAVIOR=prefer-cached
    EOF
    mkdir -p ${local.model_cache_host_path}
    chown 1001:1001 ${local.model_cache_host_path}
  EOT
}

resource "aws_launch_template" "gpu" {
  name_prefix   = "${local.name}-gpu-"
  image_id      = data.aws_ssm_parameter.ecs_gpu_ami.value
  instance_type = var.gpu_instance_type
  user_data     = base64encode(local.ecs_user_data)

  iam_instance_profile {
    arn = aws_iam_instance_profile.ecs_instance.arn
  }

  vpc_security_group_ids = [aws_security_group.ecs_instances.id]

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  monitoring {
    enabled = true
  }

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = 200
      volume_type           = "gp3"
      encrypted             = true
      delete_on_termination = true
    }
  }

  tag_specifications {
    resource_type = "instance"
    tags          = merge(local.tags, { Name = "${local.name}-gpu" })
  }
}

resource "aws_autoscaling_group" "gpu" {
  name                  = "${local.name}-gpu"
  min_size              = 0
  max_size              = 2
  vpc_zone_identifier   = module.vpc.private_subnets
  protect_from_scale_in = true
  health_check_type     = "EC2"

  launch_template {
    id      = aws_launch_template.gpu.id
    version = aws_launch_template.gpu.latest_version
  }

  tag {
    key                 = "AmazonECSManaged"
    value               = "true"
    propagate_at_launch = true
  }

  lifecycle {
    ignore_changes = [desired_capacity]
  }
}

resource "aws_ecs_capacity_provider" "gpu" {
  name = "${local.name}-gpu"

  auto_scaling_group_provider {
    auto_scaling_group_arn         = aws_autoscaling_group.gpu.arn
    managed_termination_protection = "ENABLED"
    managed_draining               = "ENABLED"

    managed_scaling {
      status                    = "ENABLED"
      target_capacity           = 100
      minimum_scaling_step_size = 1
      maximum_scaling_step_size = 1
    }
  }
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name       = aws_ecs_cluster.main.name
  capacity_providers = ["FARGATE", aws_ecs_capacity_provider.gpu.name]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 1
  }
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.name}/api"
  retention_in_days = 30
}

resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name}-api"
  network_mode             = "awsvpc"
  requires_compatibilities = [local.compute.compatibility]
  cpu                      = local.compute.cpu
  memory                   = local.compute.memory
  execution_role_arn       = aws_iam_role.api_execution.arn
  task_role_arn            = aws_iam_role.api_task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  # GPU instances keep the model cache on the host; Fargate gets a large scratch disk.
  dynamic "volume" {
    for_each = local.use_gpu ? [1] : []
    content {
      name      = "model-cache"
      host_path = local.model_cache_host_path
    }
  }

  dynamic "ephemeral_storage" {
    for_each = local.use_gpu ? [] : [1]
    content {
      size_in_gib = 100
    }
  }

  container_definitions = jsonencode([merge({
    name      = "api"
    image     = "${aws_ecr_repository.api.repository_url}:${local.api_image_tag}"
    essential = true

    portMappings = [{ containerPort = 8080, protocol = "tcp" }]

    environment = [for name, value in local.api_environment : { name = name, value = value }]
    secrets     = [for name, arn in local.api_secrets : { name = name, valueFrom = arn }]

    linuxParameters = { initProcessEnabled = true }
    ulimits         = [{ name = "nofile", softLimit = 65536, hardLimit = 65536 }]
    stopTimeout     = 120

    healthCheck = {
      command     = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=5)\""]
      interval    = 30
      timeout     = 10
      retries     = 3
      startPeriod = 300
    }

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.api.name
        awslogs-region        = var.region
        awslogs-stream-prefix = "api"
      }
    }
    }, local.use_gpu ? {
    resourceRequirements = [{ type = "GPU", value = "1" }]
    mountPoints          = [{ sourceVolume = "model-cache", containerPath = "/app/.cache", readOnly = false }]
    } : {}),
    # Self-hosted embeddings (Hugging Face TEI, OpenAI-compatible /v1/embeddings). Shares the
    # task's network namespace, so the API reaches it on localhost and nothing is exposed.
    {
      name              = "embeddings"
      image             = var.embedding_server_image
      essential         = true
      memoryReservation = 1024
      command = [
        "--model-id", var.embedding_model,
        "--port", tostring(local.embedding_port),
        "--auto-truncate",
      ]
      portMappings = [{ containerPort = local.embedding_port, protocol = "tcp" }]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.api.name
          awslogs-region        = var.region
          awslogs-stream-prefix = "embeddings"
        }
      }
    },
  ])
}

resource "aws_ecs_service" "api" {
  name                              = "paperlens-api"
  cluster                           = aws_ecs_cluster.main.id
  task_definition                   = aws_ecs_task_definition.api.arn
  desired_count                     = local.api_image_tag == "unset" ? 0 : 1
  health_check_grace_period_seconds = 600
  enable_execute_command            = true
  propagate_tags                    = "SERVICE"

  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  # Switching compute_mode changes the strategy, which ECS only applies with a new deployment.
  force_new_deployment = true

  capacity_provider_strategy {
    capacity_provider = local.compute.capacity_provider
    weight            = 1
  }

  network_configuration {
    subnets         = module.vpc.private_subnets
    security_groups = [aws_security_group.api_task.id]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8080
  }

  depends_on = [aws_lb_listener.https, aws_ecs_cluster_capacity_providers.main]
}
