# TrustTab-X local architecture

```mermaid
flowchart LR
  A[Public ACS 2018 source cache] --> B[Household-disjoint split manifest]
  B --> C[Fit five candidate pipelines]
  C --> D[Source-selection Brier chooses fixed expert]
  D --> E[Versioned manifest and pipeline SHA-256]
  E --> F[Independent approved-manifest digest]
  F --> G[Authenticated loopback FastAPI]
  G --> H[Validated fixed-expert prediction]
  G --> I[Source-reference drift diagnostics]
  G --> J[Admin-only delayed-label metrics]
  G --> K[Dashboard and privacy-conscious audit metadata]
```

The research-only router remains outside the serving path. The service loads only a locally trusted artifact whose digest matches an independently configured approval value. Predictions are schema-validated; operator/admin roles differ. Audit logs omit raw features, labels, probabilities and credentials. Runtime telemetry is in memory and resets on restart. The service was demonstrated locally; Docker, TLS, managed identity, retention governance and production authorization are incomplete. See [industry case study](INDUSTRIAL_CASE_STUDY.md) and [demo](LOCAL_DEMO.md).
