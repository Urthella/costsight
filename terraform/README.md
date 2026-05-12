# costsight — Terraform IaC

This directory provisions the production-path architecture documented
in [REPORT.md § 4.1](../REPORT.md#cloud-architecture-production-path).
Every box in the mermaid diagram corresponds to a real Terraform
resource here.

## Resources created

| Component | Resource |
|---|---|
| Raw CUR landing zone | `aws_s3_bucket.cur_raw` (+ lifecycle to IA after 60 days, 2-year retention, versioning on) |
| Aggregated parquet store | `aws_s3_bucket.aggregated` |
| Alerts table | `aws_dynamodb_table.alerts` (PAY_PER_REQUEST, PITR on, TTL via `expires_at`) |
| Alert fan-out | `aws_sns_topic.alerts` (+ optional email subscription) |
| Ingest Lambda | `aws_lambda_function.ingest` + IAM role + S3 trigger |
| Detection cluster | `aws_ecs_cluster.detection` + nightly EventBridge schedule |
| Dashboard cluster (optional) | `aws_ecs_cluster.dashboard` — only if `enable_dashboard_ecs = true` |

## Usage

```bash
cd terraform/
terraform init
terraform plan -var="env=dev" -var="alert_email=you@example.com"
terraform apply -var="env=dev" -var="alert_email=you@example.com"
```

Outputs include the S3 bucket names, DynamoDB table name, and the
SNS topic ARN — those are what you wire CUR ingest into (point AWS's
CUR delivery at the `cur_raw` bucket and subscribe additional
endpoints to the alerts topic).

## Regional carbon hint

`aws_region` defaults to `us-west-2` because its grid is one of the
cleanest AWS regions outside of Europe. For absolute lowest-carbon
deploys consider `eu-west-3` (Paris — French nuclear) or `eu-north-1`
(Stockholm — Swedish hydro). See the *Carbon* tab in the dashboard
for the cross-region comparison.

## Toggles

- `enable_ingest_lambda` — set to `false` if you're feeding the
  aggregated bucket by hand or from a non-AWS source.
- `enable_detection` — set to `false` if you're running the
  detector locally and only want the storage/alert infrastructure.
- `enable_dashboard_ecs` — set to `true` if you'd rather self-host
  the Streamlit dashboard on ECS Fargate instead of using
  Streamlit Cloud (free tier).

## Cost note

At the toggle defaults (`enable_dashboard_ecs = false`) the steady-
state monthly bill is **~$5/month** for a single tenant feeding 10
GB/mo of CUR. See REPORT § 4.1 for the per-component breakdown.
