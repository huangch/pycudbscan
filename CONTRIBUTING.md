# Contributing to PyCuDBSCAN

Thank you for considering contributing to PyCuDBSCAN! This document outlines the process for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/pycudbscan.git
   cd pycudbscan
   ```
3. Set up your development environment:
   ```bash
   conda env create -f environment.yml
   conda activate pycudbscan
   ```
4. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Development Guidelines

### Code Style

- For C++/CUDA: Follow the Google C++ Style Guide
- For Python: Follow PEP 8

### Pull Request Process

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes and test thoroughly
3. Run the test suite to ensure all tests pass:
   ```bash
   cd tests
   python -m unittest discover
   ```
4. Commit your changes with clear, descriptive commit messages
5. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
6. Create a Pull Request against the main repository's `main` branch
7. Wait for code review and address any feedback

### Adding Features

If you're adding new features, please consider the following:

1. Write tests for your feature
2. Update documentation to reflect your changes
3. Add an example if it's a user-facing feature

### Reporting Bugs

When reporting bugs, please include:

1. Your operating system and version
2. Your Python and CUDA version
3. Steps to reproduce the issue
4. Expected behavior and actual behavior
5. Any error messages (full stack traces are appreciated)

## Performance Considerations

As this is a CUDA-accelerated library, performance is a key consideration:

1. Be mindful of memory transfers between host and device
2. Consider thread divergence in your CUDA kernels
3. Profile your code to identify bottlenecks
4. Consider compatibility with different CUDA architectures

## Testing Environment

Before submitting a PR, please test your changes on:

1. Multiple CUDA versions (if possible)
2. Multiple operating systems (Linux, Windows, macOS if applicable)
3. Multiple Python versions (at least 3.7 and 3.9)

## Documentation

When adding or changing features, please update:

1. Function/method docstrings
2. README.md (if applicable)
3. Example code

Thank you for your contributions!