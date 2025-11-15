# Contributing to RFP Agent

Thank you for your interest in contributing to RFP Agent! This document provides guidelines for contributions.

## How to Contribute

### Reporting Issues

- Use GitHub Issues to report bugs or suggest features
- Check existing issues before creating a new one
- Provide detailed information:
  - Steps to reproduce (for bugs)
  - Expected vs actual behavior
  - Environment details (Python version, OS, etc.)

### Submitting Changes

1. **Fork the repository**
   ```bash
   git clone https://github.com/rdwornik/rfp-agent-cognitive-planning.git
   cd rfp-agent-cognitive-planning
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clean, readable code
   - Follow existing code style
   - Add comments for complex logic
   - Update documentation as needed

4. **Test your changes**
   ```bash
   # Test imports
   python -c "import rfp_agent"
   
   # Test CLI
   python -m rfp_agent.main --help
   
   # Test functionality
   python -m rfp_agent.main build-kb --csv example_kb.csv
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

6. **Push and create a Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Development Setup

1. Install in development mode:
   ```bash
   pip install -e .
   ```

2. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest black pylint  # Optional dev tools
   ```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and small

### Example:
```python
def process_question(question: str, context: Dict[str, Any]) -> str:
    """
    Process an RFP question with context.
    
    Args:
        question: The RFP question text
        context: Additional context for processing
        
    Returns:
        Processed question string
    """
    # Implementation
    pass
```

## Testing Guidelines

- Test new features before submitting
- Ensure existing functionality still works
- Use example files for testing
- Document test procedures

## Documentation

- Update README.md for major changes
- Update USAGE.md for new features
- Add inline comments for complex code
- Update docstrings when changing function signatures

## Areas for Contribution

We welcome contributions in these areas:

- **Features**: New functionality, integrations, improvements
- **Documentation**: Tutorials, examples, clarifications
- **Testing**: Test cases, bug reports, validation
- **Performance**: Optimization, efficiency improvements
- **Bug Fixes**: Issue resolution, error handling

## Questions?

- Open a GitHub Issue with your question
- Check existing documentation
- Review closed issues for similar questions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
