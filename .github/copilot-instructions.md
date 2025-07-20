Be an expert in Python, database algorithms, and containerization technologies.
Follow Python's official documentation and PEPs for best practices in Python development.
Apply containerization best practices for Python applications (e.g., Docker, OCI containers).
Ensure code is clean, maintainable, and secure.
Use efficient database algorithms and optimize for performance.


This specific project main purpose is to modularize the autoformer codebase for improved maintainability and extensibility. Reference the document #MODULAR_AUTOFORMER_ARCHITECTURE.md, #COMPLETE_MODEL_TECHNICAL_DOCUMENTATION, #HF_MODULAR_ARCHITECTURE_DOCUMENTATION and  #TESTING_FRAMEWORK_DOCUMENTATION

# Python Best Practices Copilot Instructions

## Project Structure
- Use clear project structure: separate directories for source code, tests, docs, and config.
- Modular design: distinct files for models, services, controllers, and utilities.
- Manage configuration with environment variables.

## Coding Standards
- Add typing annotations to all functions and classes, including return types.
- Add descriptive docstrings to all functions and classes (PEP 257 convention). Update existing docstrings if needed.
- Keep all existing comments in files.
- Use Ruff for code style consistency.

## Error Handling & Logging
- Implement robust error handling and logging, including context capture.

## Testing
- Use pytest (not unittest) and pytest plugins for all tests.
- Place all tests in ./tests. Create __init__.py files as needed.
- All tests must have typing annotations and docstrings.
- Update tests as much as possible and dont create new one
- For type checking in tests, import:
  ```python
  from _pytest.capture import CaptureFixture
  from _pytest.fixtures import FixtureRequest
  from _pytest.logging import LogCaptureFixture
  from _pytest.monkeypatch import MonkeyPatch
  from pytest_mock.plugin import MockerFixture
  ```

## Documentation & Dependency Management
- Use docstrings and README files for documentation.
- Use https://github.com/astral-sh/uv and virtual environments for dependency management.

## CI/CD
- Use GitHub Actions or GitLab CI for CI/CD.

## AI-Friendly Coding
- Provide code snippets and explanations optimized for clarity and AI-assisted development.

## Remove reduntant code, files, tests and documents
- Identify and remove any redundant code, files, tests, and documentation to streamline the project.