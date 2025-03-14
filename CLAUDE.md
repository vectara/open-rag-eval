# Vectara-Eval Guidelines

## Commands
- Run all tests: `python -m unittest discover`
- Run single test: `python -m unittest path/to/test_file.py`
- Run specific test case: `python -m unittest path/to/test_file.py TestClassName.test_method_name`
- Run evaluation: `python run_eval.py`
- Run server: `python run_server.py`

## Code Style Guidelines
- **Imports**: Standard library first, third-party next, local last (alphabetical in each group)
- **Types**: Use Python type hints for all function parameters and return values
- **Classes**: PascalCase (VectaraConnector), implement abstract base classes when appropriate
- **Functions/Variables**: snake_case for methods, functions, variables
- **Constants**: UPPER_SNAKE_CASE
- **Privacy**: Prefix private members with underscore (_api_key)
- **Error Handling**: Use specific exceptions, provide context when re-raising
- **Documentation**: Docstrings for classes/functions with Args/Returns/Raises sections
- **File Structure**: Group related functionality in dedicated modules