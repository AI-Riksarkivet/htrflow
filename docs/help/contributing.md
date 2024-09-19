# Contributing to HTRflow 🛠️

Thank you for your interest in contributing to HTRflow! We appreciate contributions in the following areas:

1. **New Features**: Enhance the library by adding new functionality. Refer to the section below for guidelines.
2. **Documentation**: Help us improve our documentation with clear examples demonstrating how to use HTRflow.
3. **Bug Reports**: Identify and report any issues in the project.
4. **Feature Requests**: Suggest new features or improvements.

## Contributing Features

HTRflow is developed as an open source project, were we value contributions that offer generic solutions to common problems. Before proposing a new feature, please open an issue to discuss your idea with the community. This encourages feedback and support.

## How to Contribute

Develop your feature, fix, or documentation update on your branch.

### Code Quality 🎨

Ensure your code adheres to our quality standards using tools like:

```bash
ruff
```


### Documentation 📝

Our documentation utilizes docstrings combined with type hinting from mypy. Update or add necessary documentation in the `docs/` directory and test it locally with:

   ```bash
   mkdocs serve -v
   ```

### Tests 🧪

We employ pytest for testing. Ensure you add tests for your changes and run:

   ```bash
   pytest
   ```

## Making a Pull Request

After pushing your changes to GitHub, initiate a pull request from your fork to the main `HTRflow` repository:

Push your branch:

```bash
git push -u origin <your_branch_name>
```

Visit the repository on GitHub and click "New Pull Request." Set the base branch to `develop` and describe your changes.

Ensure all tests pass before requesting a review.
