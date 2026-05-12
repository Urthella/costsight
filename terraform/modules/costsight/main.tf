# costsight — root module implementation.
# All resources are conditionally created via count = X ? 1 : 0 so the
# stack can be brought up incrementally (dev → staging → prod).

locals {
  name_prefix = "costsight-${var.env}"
}

# ---------------------------------------------------------------------------
# S3 — Raw CUR landing zone + aggregated parquet store
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "cur_raw" {
  bucket = "${local.name_prefix}-cur-raw"
  tags   = var.tags
}

resource "aws_s3_bucket_lifecycle_configuration" "cur_raw" {
  bucket = aws_s3_bucket.cur_raw.id

  rule {
    id     = "transition-cold-cur-to-ia"
    status = "Enabled"

    transition {
      days          = 60
      storage_class = "STANDARD_IA"
    }

    expiration {
      days = 730 # 2 years of CUR retention covers most compliance regimes.
    }
  }
}

resource "aws_s3_bucket_versioning" "cur_raw" {
  bucket = aws_s3_bucket.cur_raw.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket" "aggregated" {
  bucket = "${local.name_prefix}-aggregated"
  tags   = var.tags
}

# ---------------------------------------------------------------------------
# DynamoDB — alerts table
# ---------------------------------------------------------------------------

resource "aws_dynamodb_table" "alerts" {
  name         = "${local.name_prefix}-alerts"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "alert_id"
  range_key    = "alert_date"

  attribute {
    name = "alert_id"
    type = "S"
  }

  attribute {
    name = "alert_date"
    type = "S"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }

  point_in_time_recovery {
    enabled = true
  }

  tags = var.tags
}

# ---------------------------------------------------------------------------
# SNS — alert fan-out topic
# ---------------------------------------------------------------------------

resource "aws_sns_topic" "alerts" {
  name = "${local.name_prefix}-alerts"
  tags = var.tags
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.alert_email == "" ? 0 : 1
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# ---------------------------------------------------------------------------
# Ingest Lambda — fires on S3 PutObject in cur_raw bucket
# ---------------------------------------------------------------------------

resource "aws_iam_role" "ingest_lambda" {
  count = var.enable_ingest_lambda ? 1 : 0
  name  = "${local.name_prefix}-ingest-lambda"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action = "sts:AssumeRole"
    }]
  })

  tags = var.tags
}

resource "aws_iam_role_policy" "ingest_lambda_inline" {
  count = var.enable_ingest_lambda ? 1 : 0
  name  = "ingest-lambda-policy"
  role  = aws_iam_role.ingest_lambda[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.cur_raw.arn,
          "${aws_s3_bucket.cur_raw.arn}/*",
        ]
      },
      {
        Effect = "Allow"
        Action = ["s3:PutObject"]
        Resource = ["${aws_s3_bucket.aggregated.arn}/*"]
      },
      {
        Effect = "Allow"
        Action = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:*:*:*"
      },
    ]
  })
}

# Placeholder for the actual Lambda artifact. In a real account, the
# artifact would be built from src/cloud_anomaly/preprocessing.py and
# pushed to S3 as a deployment package. Here we leave it as a stub so
# `terraform plan` succeeds without a hard dependency on a built zip.
resource "aws_lambda_function" "ingest" {
  count         = var.enable_ingest_lambda ? 1 : 0
  function_name = "${local.name_prefix}-ingest"
  role          = aws_iam_role.ingest_lambda[0].arn
  handler       = "preprocessing.lambda_handler"
  runtime       = "python3.11"
  timeout       = 300
  memory_size   = 1024

  # Build artifact: replace with a real .zip in production.
  filename         = "${path.module}/placeholder.zip"
  source_code_hash = filebase64sha256("${path.module}/placeholder.zip")

  environment {
    variables = {
      AGGREGATED_BUCKET = aws_s3_bucket.aggregated.id
      ALERTS_TABLE      = aws_dynamodb_table.alerts.name
    }
  }

  tags = var.tags
}

resource "aws_s3_bucket_notification" "cur_raw" {
  count  = var.enable_ingest_lambda ? 1 : 0
  bucket = aws_s3_bucket.cur_raw.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.ingest[0].arn
    events              = ["s3:ObjectCreated:*"]
    filter_suffix       = ".csv.gz"
  }
}

resource "aws_lambda_permission" "ingest_s3" {
  count         = var.enable_ingest_lambda ? 1 : 0
  statement_id  = "AllowS3Invoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ingest[0].function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.cur_raw.arn
}

# ---------------------------------------------------------------------------
# Detection — ECS Fargate task triggered nightly by EventBridge
# ---------------------------------------------------------------------------

resource "aws_ecs_cluster" "detection" {
  count = var.enable_detection ? 1 : 0
  name  = "${local.name_prefix}-detection"
  tags  = var.tags

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_cloudwatch_event_rule" "nightly_detection" {
  count               = var.enable_detection ? 1 : 0
  name                = "${local.name_prefix}-nightly-detection"
  description         = "Trigger the costsight detection pass once per day."
  schedule_expression = "cron(0 4 * * ? *)"  # 04:00 UTC daily
  tags                = var.tags
}

# (ECS task definition + EventBridge target wiring would follow here in
# a production module — left out to keep the placeholder buildable.)

# ---------------------------------------------------------------------------
# Optional dashboard ECS service
# ---------------------------------------------------------------------------

resource "aws_ecs_cluster" "dashboard" {
  count = var.enable_dashboard_ecs ? 1 : 0
  name  = "${local.name_prefix}-dashboard"
  tags  = var.tags
}
