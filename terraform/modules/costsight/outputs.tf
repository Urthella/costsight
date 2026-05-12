output "cur_raw_bucket" {
  description = "S3 bucket where AWS delivers the CUR feed."
  value       = aws_s3_bucket.cur_raw.id
}

output "aggregated_bucket" {
  description = "S3 bucket holding the preprocessed Parquet partitions."
  value       = aws_s3_bucket.aggregated.id
}

output "alerts_table" {
  description = "DynamoDB table receiving severity-banded alerts."
  value       = aws_dynamodb_table.alerts.name
}

output "alerts_topic" {
  description = "SNS topic ARN; subscribe Slack / email / PagerDuty here."
  value       = aws_sns_topic.alerts.arn
}

output "dashboard_url" {
  description = "Public URL of the Streamlit dashboard (only if dashboard_ecs enabled)."
  value       = var.enable_dashboard_ecs ? "to-be-populated-after-load-balancer-attaches" : "(dashboard_ecs disabled — use Streamlit Cloud)"
}
