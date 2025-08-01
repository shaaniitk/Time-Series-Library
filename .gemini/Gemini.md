# AI Rules for {{project-name}}

{{project-description}}

## CODING_PRACTICES

### Guidelines for SUPPORT_LEVEL

#### SUPPORT_EXPERT

- Favor elegant, maintainable solutions over verbose code. Assume understanding of language idioms and design patterns.
- Highlight potential performance implications and optimization opportunities in suggested code.
- Frame solutions within broader architectural contexts and suggest design alternatives when appropriate.
- Focus comments on 'why' not 'what' - assume code readability through well-named functions and variables.
- Proactively address edge cases, race conditions, and security considerations without being prompted.
- When debugging, provide targeted diagnostic approaches rather than shotgun solutions.
- Suggest comprehensive testing strategies rather than just example tests, including considerations for mocking, test organization, and coverage.


### Guidelines for DOCUMENTATION

#### SWAGGER

- Define comprehensive schemas for all request and response objects
- Use semantic versioning in API paths to maintain backward compatibility
- Implement detailed descriptions for endpoints, parameters, and {{domain_specific_concepts}}
- Configure security schemes to document authentication and authorization requirements
- Use tags to group related endpoints by resource or functional area
- Implement examples for all endpoints to facilitate easier integration by consumers


### Guidelines for ARCHITECTURE

#### ADR

- Create ADRs in /docs/adr/{name}.md for:
- 1) Major dependency changes
- 2) Architectural pattern changes
- 3) New integration patterns
- 4) Database schema changes

#### CLEAN_ARCHITECTURE

- Strictly separate code into layers: entities, use cases, interfaces, and frameworks
- Ensure dependencies point inward, with inner layers having no knowledge of outer layers
- Implement domain entities that encapsulate {{business_rules}} without framework dependencies
- Use interfaces (ports) and implementations (adapters) to isolate external dependencies
- Create use cases that orchestrate entity interactions for specific business operations
- Implement mappers to transform data between layers to maintain separation of concerns

#### DDD

- Define bounded contexts to separate different parts of the domain with clear boundaries
- Implement ubiquitous language within each context to align code with business terminology
- Create rich domain models with behavior, not just data structures, for {{core_domain_entities}}
- Use value objects for concepts with no identity but defined by their attributes
- Implement domain events to communicate between bounded contexts
- Use aggregates to enforce consistency boundaries and transactional integrity


### Guidelines for VERSION_CONTROL

#### GIT

- Use conventional commits to create meaningful commit messages
- Use feature branches with descriptive names following {{branch_naming_convention}}
- Write meaningful commit messages that explain why changes were made, not just what
- Keep commits focused on single logical changes to facilitate code review and bisection
- Use interactive rebase to clean up history before merging feature branches
- Leverage git hooks to enforce code quality checks before commits and pushes


## TESTING

### Guidelines for UNIT

#### PYTEST

- Use fixtures for test setup and dependency injection
- Implement parameterized tests for testing multiple inputs for {{function_types}}
- Use monkeypatch for mocking dependencies


### Guidelines for E2E

#### SELENIUM

- Use the page object model to organize test code
- Implement explicit waits instead of implicit waits for {{dynamic_elements}}
- Use the WebDriverWait class for waiting for elements to be ready


### Guidelines for INTEGRATION

#### SUPERTEST

- Use async/await with supertest for cleaner test code
- Implement a test database for integration tests
- Use beforeEach/afterEach hooks for database setup and teardown when testing {{api_endpoints}}


## BACKEND

### Guidelines for PYTHON

#### FASTAPI

- Use Pydantic models for request and response validation with strict type checking and custom validators
- Implement dependency injection for services and database sessions to improve testability and resource management
- Use async endpoints for I/O-bound operations to improve throughput for {{high_load_endpoints}}
- Leverage FastAPI's background tasks for non-critical operations that don't need to block the response
- Implement proper exception handling with HTTPException and custom exception handlers for {{error_scenarios}}
- Use path operation decorators consistently with appropriate HTTP methods (GET for retrieval, POST for creation, etc.)


