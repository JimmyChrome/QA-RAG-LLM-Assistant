# Contributing

Thank you for your interest in contributing to this project!

This repository aims to develop an open-source Retrieval-Augmented Generation (RAG) chatbot for the University of the Philippines Diliman Quality Assurance Office. Contributions that improve code quality, documentation, testing, and system design are welcome.

## Before You Start

Please:

- Read the README.
- Check existing Issues before creating a new one.
- Open an Issue first if you plan to make a major change.

## Development Setup

1. Sample
2. Sample

```bash
git clone https://github.com/<username>/<repository>.git
```

3. Sample

```bash
python -m venv .venv
```

4. Sample

```bash
pip install -r requirements.txt
```

5. Sample

```bash
cp .env.example .env
```

6. Sample

## Branch Naming

Please use descriptive branch names.

Examples:

```
feature/document-upload
feature/chunking
feature/vector-search
fix/pdf-parser
docs/update-readme
refactor/retrieval
```

## Commit Messages

Use the following convention:

```
Git Commit Pattern:
<type>: <change>

feat: new feature for the user, not a new feature for build script
fix: bug fix for the user, not a fix to a build script
docs: changes to the documentation
style: formatting, missing semi colons, etc; no production code change
refactor: refactoring production code, eg. renaming a variable
test: adding missing tests, refactoring tests; no production code change
chore: updating grunt tasks etc; no production code change

Rules: 
- Commit as atomic as possible

- Create a branch for every feature with the pattern feature/<title> by checking out from main
 
e.g. feature/create-task, feature/task-store

- Always push your local branch to the github repository
say you have feature/create-task, i'd like to see that pushed in feature/create-task branch in the repository 
- You can do this via git push origin head
- Create your own pull request once you finish a feature / goal for a branch
- This ensures that pull requests get credited to the right person (add comments, revisions etc.)

```

## Pull Requests

Each Pull Request should:

- Focus on one feature or fix.
- Include documentation updates when applicable.
- Include tests whenever possible.
- Pass all automated checks.

Please describe:

- What changed
- Why it changed
- How it was tested

## Coding Style

General guidelines:

- Use descriptive variable and function names.
- Write docstrings for public functions.
- Avoid unnecessary complexity.
- Keep functions small and focused.

## Documentation

If your contribution changes the architecture, APIs, or workflows, please update the relevant documentation.

## Testing

New features should include appropriate tests whenever practical.

## Questions

If you're unsure about an implementation, feel free to open an Issue for discussion before beginning development.

Thank you for contributing!