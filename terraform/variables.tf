variable "env" {
  description = "Deployment environment (dev / staging / prod). Used as a suffix in resource names."
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region for all primary resources. Pick a low-carbon region (us-west-2, eu-west-3, eu-north-1) when possible."
  type        = string
  default     = "us-west-2"
}

variable "alert_email" {
  description = "Optional email address to subscribe to the SNS alerts topic. Leave blank to skip the subscription."
  type        = string
  default     = ""
}

variable "enable_ingest_lambda" {
  description = "Provision the ingest Lambda (S3 PutObject → preprocess)."
  type        = bool
  default     = true
}

variable "enable_detection" {
  description = "Provision the ECS Fargate detection task + scheduled CloudWatch rule."
  type        = bool
  default     = true
}

variable "enable_dashboard_ecs" {
  description = "Provision an ECS Fargate service for the Streamlit dashboard. Streamlit Cloud is the cheaper / recommended default; keep this off unless you need self-hosted."
  type        = bool
  default     = false
}
