# Contributing

1. Open an issue describing the bug or feature (use the templates).
2. Create a branch: `feature/<short-name>` or `fix/<short-name>`.
3. Write tests first or together with the code; keep coverage at 90 % or more.
4. Run `ruff check . && ruff format . && pytest` locally.
5. Use clear commit messages: `feat: ...`, `fix: ...`, `test: ...`, `docs: ...`, `ci: ...`.
6. Open a pull request to `main`; it is merged only when CI is green.

Releases: update `CHANGELOG.md` and the version in `pyproject.toml` and
`src/vadeval/__init__.py`, then push a tag `vX.Y.Z`.
