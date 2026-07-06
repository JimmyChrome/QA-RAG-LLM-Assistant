# Security Policy

## Supported Versions

Security updates are provided for the latest development version of the project.

Older versions may not receive security fixes.

## Reporting a Vulnerability

Please do not disclose security vulnerabilities publicly.

Instead:

1. Contact the project maintainers privately.
2. Provide a detailed description of the issue.
3. Include reproduction steps if possible.
4. Allow reasonable time for investigation before public disclosure.

## Scope

Examples of security issues include:

- Authentication bypass
- Authorization issues
- Remote code execution
- Injection vulnerabilities
- Sensitive information exposure
- Dependency vulnerabilities

## Sensitive Data

This project is designed to work with institutional documents.

Contributors should **never**:

- Commit confidential documents.
- Commit generated embeddings from confidential documents.
- Commit vector databases containing confidential information.
- Commit API keys, passwords, or secrets.

Use `.env` files for configuration and ensure they are excluded from version control.

## Responsible Disclosure

We appreciate responsible disclosure of vulnerabilities.

Once verified, maintainers will investigate, develop a fix, and acknowledge the reporter where appropriate.