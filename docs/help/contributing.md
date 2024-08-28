# Contributing to htrflow 🛠️

Thank you for your interest in contributing to htrflow! We appreciate contributions in the following areas:

1. **New Features**: Enhance the library by adding new functionality. Refer to the section below for guidelines.
2. **Documentation**: Help us improve our documentation with clear examples demonstrating how to use htrflow.
3. **Bug Reports**: Identify and report any issues in the project.
4. **Feature Requests**: Suggest new features or improvements.

## Contributing Features ✨

htrflow aims to provide versatile tools applicable across a broad range of projects. We value contributions that offer generic solutions to common problems. Before proposing a new feature, please open an issue to discuss your idea with the community. This encourages feedback and support.

## How to Contribute

1. Fork the htrflow repository to your GitHub account by clicking "fork" at the top right of the repository page.
2. Clone your fork locally and create a new branch for your changes:

   ```bash
   git clone https://github.com/yourusername/htrflow.git
   cd htrflow
   git checkout -b <your_branch_name>
   ```

3. Develop your feature, fix, or documentation update on your branch.

### Code Quality 🎨

Ensure your code adheres to our quality standards using tools like:

- ruff
- mypy

### Documentation 📝

Our documentation utilizes docstrings combined with type hinting from mypy. Update or add necessary documentation in the `docs/` directory and test it locally with:

   ```bash
   mkdocs serve -v
   ```

## Tests 🧪

We employ pytest for testing. Ensure you add tests for your changes and run:

   ```bash
   pytest
   ```

## Making a Pull Request

After pushing your changes to GitHub, initiate a pull request from your fork to the main `htrflow` repository:

1. Push your branch:

   ```bash
   git push -u origin <your_branch_name>
   ```

2. Visit the repository on GitHub and click "New Pull Request." Set the base branch to `develop` and describe your changes.

Ensure all tests pass before requesting a review.

## License 📄

By contributing to htrflow, you agree that your contributions will be licensed under the [EUPL-1.2 license](https://github.com/AI-Riksarkivet/htrflow/tree/main?tab=EUPL-1.2-1-ov-file#readme).

Thank you for contributing to htrflow!