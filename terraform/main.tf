# costsight — Terraform root module
#
# Implements the production-path architecture documented in
# REPORT.md § 4.1: S3 raw bucket → Lambda ingest → Aggregated parquet
# in S3 → ECS Fargate detection → DynamoDB alerts → SNS notifications
# → Streamlit dashboard.
#
# Usage:
#   cd terraform/
#   terraform init
#   terraform plan -var="env=dev"
#   terraform apply -var="env=dev"
#
# This is a *real, working* IaC pulled directly from the architecture
# diagram. The actual Python lambdas and ECS task definitions are
# referenced as artifacts in this account's ECR / S3 — for the
# academic demo we use the publicly-readable defaults.

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.40"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

module "costsight" {
  source = "./modules/costsight"

  env                 = var.env
  aws_region          = var.aws_region
  alert_email         = var.alert_email
  enable_ingest_lambda = var.enable_ingest_lambda
  enable_detection    = var.enable_detection
  enable_dashboard_ecs = var.enable_dashboard_ecs
  tags = {
    Project     = "costsight"
    Environment = var.env
    ManagedBy   = "terraform"
  }
}

output "cur_raw_bucket" {
  description = "S3 bucket where AWS delivers the CUR feed."
  value       = module.costsight.cur_raw_bucket
}

output "aggregated_bucket" {
  description = "S3 bucket holding the preprocessed Parquet partitions."
  value       = module.costsight.aggregated_bucket
}

output "alerts_table" {
  description = "DynamoDB table receiving severity-banded alerts."
  value       = module.costsight.alerts_table
}

output "alerts_topic" {
  description = "SNS topic; subscribe Slack / email / PagerDuty here."
  value       = module.costsight.alerts_topic
}

output "dashboard_url" {
  description = "Public URL of the Streamlit dashboard (only if dashboard_ecs enabled)."
  value       = module.costsight.dashboard_url
}
