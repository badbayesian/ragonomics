# Cloud Deployment Notes (WIP)

This document outlines cloud deployment patterns for Ragonometrics and highlights
certification-aligned competencies (AWS/GCP) through concrete infrastructure design.

Target Architecture (AWS)
-------------------------
- Compute: ECS/Fargate service for API + Streamlit UI
- Data: RDS Postgres for metadata + vector text
- Cache/Queue: ElastiCache Redis for RQ workers
- Storage: S3 for artifacts (indexes, manifests, reports)
- Observability: CloudWatch logs/metrics, structured app logs

Target Architecture (GCP)
-------------------------
- Compute: Cloud Run or GKE for API + Streamlit UI
- Data: Cloud SQL Postgres
- Cache/Queue: Memorystore Redis
- Storage: GCS for artifacts
- Observability: Cloud Logging + Cloud Monitoring

Secrets and Config
------------------
- Store API keys in Secrets Manager/Secret Manager.
- Inject at runtime using task definitions or workload identity.
- Keep local environment configuration for dev only (see [Configuration](https://github.com/badbayesian/ragonometrics/blob/main/docs/configuration/configuration.md)).

Deployment Checklist
--------------------
- Provision Postgres + Redis with private networking.
- Create an artifact bucket for index shards and run manifests.
- Build/push images (UI + worker) to ECR/GCR.
- Apply network policies to restrict DB access to services.
- Enable structured logs and metrics at the platform layer.

Cert Alignment (Examples)
-------------------------
- AWS Certified Data Engineer / ML Engineer:
  - Data stores, orchestration, and model-serving topology.
  - Secure IAM access, private subnets, and auditing.
- GCP Professional Data Engineer:
  - Cloud SQL + Memorystore + GCS integration patterns.
  - Observability and CI/CD flows.

References in Repo
------------------
- [`deploy/terraform/aws/`](https://github.com/badbayesian/ragonometrics/tree/main/deploy/terraform/aws) and [`deploy/terraform/gcp/`](https://github.com/badbayesian/ragonometrics/tree/main/deploy/terraform/gcp) stubs
- [`docs/architecture/architecture.md`](https://github.com/badbayesian/ragonometrics/blob/main/docs/architecture/architecture.md) for component overview
