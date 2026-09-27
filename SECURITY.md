# Security Policy

## Supported Versions

As this project is currently an active hackathon research prototype, security patches are applied directly to the primary branch (`main`).

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x (Prototype) | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability or sensitive data leak in this repository:

1. **Do not create a public issue.**
2. Send an email to the repository maintainers with a detailed description of the vulnerability, steps to reproduce, and potential impact.
3. Maintainers will review the submission within 48 hours and coordinate remediation.

## Credentials, Secrets, and Environment Variables

- **Zero Secrets in Git:** Never commit secrets, authentication tokens, cloud credentials, database connection strings, or `.env` files to this repository.
- **Environment Templates:** Use `.env.example` as a template for local development configuration.
- **Git Hooks & Scanning:** Contributors are encouraged to use pre-commit secret scanners (e.g., `git-secrets`, `trufflehog`) before pushing branches.

## Data Privacy & Public Repository Boundaries

- **No Proprietary Identifiers:** Real customer names, personal phone numbers, or private payment records must never be added to datasets or mock tests.
- **Local Data Storage:** All raw delivery telemetry and processed datasets (CSV / Parquet) are strictly ignored by `.gitignore` and must reside only on local environments.
- **Public Datasets Policy:** Only synthetic, anonymized, or explicitly approved open sample datasets may be shared.
