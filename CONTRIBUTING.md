# Contributing

## Development setup

1. Clone the repository:
   ```bash
   git clone https://github.com/wildedge/wildedge-python.git
   cd wildedge-python
   ```

2. Install dependencies:
   ```bash
   uv sync --group dev
   ```

3. Run tests:
   ```bash
   uv run pytest
   ```

4. Run tests across Python versions:
   ```bash
   uv run tox
   ```

## Code style

- Use `ruff` for linting and formatting:
  ```bash
  uv run ruff check .
  uv run ruff format .
  ```

- Follow PEP 8 and PEP 484 (type hints).

## Pull requests

1. Fork the repository and create a feature branch off `devel`.
2. Make your changes and ensure tests pass.
3. Update documentation if needed.
4. Submit a pull request with a clear description of the changes.

## Reporting issues

- Use GitHub Issues for bugs and feature requests.
- Include steps to reproduce, Python version, OS, and relevant logs.

## License

This project is licensed under the [Business Source License 1.1](LICENSE). By contributing, you agree that your contributions will be licensed under the same terms.
