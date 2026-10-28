# OpenProb development

- Public project: openbook-data-standards/openprob. Experimental Python contracts, not a published standard. Do not change existing OpenBook-family repositories as part of this work.
- Provider-neutral code and synthetic data only. Vendor integrations, private models, weights, feed payloads and credentials belong in private implementations.
- Python 3.11+, standard library at runtime. Lint: `ruff check src tests`; formatting: `ruff format --check src tests`.
- Local synthetic demonstration: `PYTHONPATH=src python3 -B -m openprob`. No network or storage is needed.
- Do not run test runners locally. Tests are authored in tests/ and need authorized CI setup before remote execution.
- No deployment, recurring jobs, trading or public model-code execution. Do not push or open PRs unless requested. Once the empty repository is bootstrapped, the line is main; future work branches from freshly fetched origin/main.
