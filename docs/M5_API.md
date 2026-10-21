# M5 API contract

Launch the local service as shown in [README](../README.md). Interactive schema and examples are available at `/docs`; OpenAPI JSON is at `/openapi.json`.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Startup readiness and loaded model version |
| `GET /model` | Task, fixed expert, schema, target mapping and research decision |
| `GET /metrics` | In-memory local request/error/latency aggregates |
| `POST /predict` | One ACSIncome record |
| `POST /predict-batch` | 1–100 records, same fixed online policy |
| `POST /monitor/drift` | Source-reference comparison for 1–1000 records |
| `POST /monitor/delayed-labels` | Offline realised metrics for aligned labels and probabilities |
| `GET /` | Local analytics interface |

Example PowerShell request (valid category codes are strings, including `.0` where present in the source):

```powershell
$body = @{
  schema_version = 'acs-income-2018-v1'
  record = @{ AGEP=39; WKHP=40; COW='1.0'; SCHL='21.0'; MAR='1'; OCCP='2310.0'; POBP='42'; RELP='0'; SEX='1'; RAC1P='1' }
} | ConvertTo-Json -Depth 4
Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body $body
```

The response includes `decision_id`, UTC timestamp, `model_version=m5-acs-fixed-1`, `schema_version`, `selected_expert`, `probability_positive`, `predicted_class`, `decision_threshold=0.5`, `abstention=false`, and a reliability warning. The probability is deterministic for the same loaded artifact and record; audit ID/time vary. Missing fields, invalid ranges, unsupported schema, unexpected categories and oversized batches return HTTP 422. Unknown categories are rejected even though the internal encoder can technically ignore them. Raw input fields are not logged.
