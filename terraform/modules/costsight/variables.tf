variable "env" {
  description = "Environment suffix (dev / staging / prod)."
  type        = string
}

variable "aws_region" {
  description = "AWS region — prefer low-carbon (us-west-2 / eu-west-3 / eu-north-1) where workload latency allows."
  type        = string
}

variable "alert_email" {
  description = "Optional email to subscribe to the SNS alerts topic."
  type        = string
  default     = ""
}

variable "enable_ingest_lambda" {
  description = "Provision the ingest Lambda + S3 trigger."
  type        = bool
  default     = true
}

variable "enable_detection" {
  description = "Provision the ECS Fargate detection task + scheduled rule."
  type        = bool
  default     = true
}

variable "enable_dashboard_ecs" {
  description = "Provision an ECS Fargate service for the Streamlit dashboard."
  type        = bool
  default     = false
}

variable "tags" {
  description = "Resource tags applied to every resource."
  type        = map(string)
  default     = {}
}
