# Contributing to Comet Fragmentation Simulator

Thank you for your interest in contributing to the Comet Fragmentation Simulator! This project aims to provide accurate, educational, and accessible astrophysics simulations.

## 🌟 Ways to Contribute

### 🐛 Bug Reports
- Use the GitHub issue tracker
- Include Python version, OS, and error messages
- Provide minimal reproduction steps

### 🚀 Feature Requests
- Check existing issues first
- Describe the scientific motivation
- Include relevant physics equations or references

### 📝 Code Contributions
- Fork the repository
- Create a feature branch (`git checkout -b feature/amazing-physics`)
- Follow the code style guidelines below
- Add tests for new functionality
- Update documentation

### 📚 Documentation
- Improve README clarity
- Add docstrings to functions
- Create tutorials or examples
- Fix typos and grammar

## 🔬 Code Style Guidelines

### Python Standards
- Follow PEP 8 style guide
- Use type hints where possible
- Maximum line length: 88 characters (Black formatter)
- Use meaningful variable and function names

### Physics Code
- Include units in docstrings and comments
- Reference scientific sources in comments
- Use SI units throughout calculations
- Add physical validation checks

### Documentation
- Write clear docstrings for all public functions
- Include parameter descriptions and units
- Provide usage examples
- Reference scientific literature when applicable

## 🧪 Testing

### Running Tests
```bash
python -m pytest tests/
```

### Test Coverage
- Aim for >80% test coverage
- Include physics validation tests
- Test edge cases and error conditions

### Physics Validation
- Compare results with known analytical solutions
- Validate energy conservation
- Check dimensional analysis

## 📦 Development Setup

### Environment Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/comet-fragmentation-simulator.git
cd comet-fragmentation-simulator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .
```

### Pre-commit Hooks
```bash
pre-commit install
```

## 🔬 Scientific Accuracy

This project prioritizes scientific accuracy:

- **Cite Sources**: Include references for physics equations
- **Validate Results**: Compare with observations or analytical solutions
- **Document Assumptions**: Clearly state model limitations
- **Peer Review**: Complex physics changes should be reviewed by domain experts

## 📋 Pull Request Process

1. **Fork & Branch**: Create a feature branch from main
2. **Develop**: Make your changes with tests and documentation
3. **Test**: Ensure all tests pass and physics validation succeeds
4. **Document**: Update relevant documentation
5. **Submit**: Create a pull request with clear description

### Pull Request Checklist
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] Physics validation included
- [ ] Code follows style guidelines
- [ ] Scientific references included (if applicable)

## 🤝 Code of Conduct

### Our Standards
- **Respectful**: Be respectful and inclusive
- **Collaborative**: Focus on constructive feedback
- **Scientific**: Value accuracy and evidence-based discussion
- **Educational**: Help others learn and understand

### Enforcement
Unacceptable behavior can be reported to the project maintainers.

## 🏷️ Release Process

1. **Version Bump**: Update version in `__init__.py`
2. **Changelog**: Update `CHANGELOG.md`
3. **Tag**: Create git tag (`v1.0.0`)
4. **Release**: GitHub release with notes

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Documentation**: Check the `docs/` directory

## 🙏 Recognition

Contributors will be:
- Listed in `CONTRIBUTORS.md`
- Mentioned in release notes
- Credited in scientific publications (if applicable)

Thank you for helping make astrophysics simulation more accessible! 🌌
