# Security Policy

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | ✅ Yes             |
| < 1.0   | ❌ No              |

## Reporting a Vulnerability

The Comet Fragmentation Simulator project takes security seriously. If you discover a security vulnerability, please follow these guidelines:

### 🔒 For Security Issues

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please:

1. **Email**: Send details to [security@example.com] (replace with actual email)
2. **Include**: 
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if known)

### ⚡ For General Issues

For non-security bugs and issues:
- Use GitHub Issues
- Follow the issue template
- Include reproduction steps

### 🛡️ Security Considerations

This project involves:
- **Scientific Computing**: Numerical calculations and data processing
- **File I/O**: Reading configuration and writing results
- **Visualization**: Generating plots and reports

### 🔍 Common Security Areas

While this is primarily a scientific simulation tool, be aware of:

- **Input Validation**: Large numerical values or malformed data
- **File Operations**: Ensure safe file paths and permissions
- **Dependencies**: Keep scientific libraries updated
- **Memory Usage**: Large simulations may consume significant resources

### 📋 Security Best Practices

When using this software:

1. **Validate Inputs**: Check simulation parameters for reasonable ranges
2. **Resource Limits**: Monitor memory and CPU usage for large simulations
3. **File Permissions**: Ensure output directories have appropriate permissions
4. **Dependencies**: Keep Python packages updated

### 🔄 Update Process

Security updates will be:
- Released as patch versions (e.g., 1.0.1)
- Documented in CHANGELOG.md
- Announced in GitHub releases

### 🤝 Coordinated Disclosure

We follow responsible disclosure principles:
- 90-day disclosure timeline
- Coordination with package maintainers
- Credit to security researchers (with permission)

### 📞 Contact Information

- **Security Issues**: [security@example.com]
- **General Issues**: GitHub Issues
- **Project Maintainers**: Listed in CONTRIBUTORS.md

Thank you for helping keep the Comet Fragmentation Simulator secure! 🌌
