# Web Console and Automation Architecture

The web console does not replace the research workflow. It packages the same validated steps into a reviewable, human-in-the-loop execution path.

```text
Browser upload
-> API creates a run record
-> worker launches one pipeline stage at a time
-> each stage writes artifacts, stage.json, and stage.log
-> browser shows stage-scoped videos and files
-> automatic WBT validation
-> human approval
-> queued GPU training
```

## Components

| Component | Responsibility |
|---|---|
| `web/` | Upload, job status, per-stage result pages, manual approval action |
| `server/` | HTTP API, persistent job state, file delivery, queue-facing endpoints |
| `g1pipe/` | GVHMR, GMR, ProtoMotions/PyRoki, WBT conversion and validation orchestration |
| `configs/` | Public example configuration only; machine-specific paths stay local |
| `runs/` | Local runtime evidence; excluded from source control except selected portfolio demos |

## Human approval and GPU safety

Automated conversion is allowed to reach a validated WBT motion, but training is a separate state transition. The operator reviews the visual evidence and validation report, then explicitly approves the run. A queue serializes training jobs so GPU-heavy work does not collide with interactive preview or remote desktop use.

## Deployment boundary

For a portfolio deployment, host the UI, job metadata, and curated media on personal infrastructure. Keep the GPU worker private and have it poll the queue outbound. This separates public viewing traffic from the machine that runs research environments and training workloads.

## Relationship to manual reproduction

Each automated stage has a manual counterpart documented in [Manual Reproduction Guide](manual-reproduction.md). This is deliberate: the web application is an orchestration layer around verified research commands, not an opaque black box.
