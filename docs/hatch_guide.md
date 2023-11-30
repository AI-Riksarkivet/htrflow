# Quick tutorial on Hatch usage

## Configuration
Hatch can be configured in [tool.hatch] blocks in pyproject.toml or in hatch.toml without the tool header. I've created a hatch.toml configuration as it allows more clarity in what is hatch configuration and what isn't and makes moving to other build systems if necessary much easier.

## Environments
Hatch has the ability to manage environments for different purposes with different packages. I've set up a test environment, which means that instead of having to install with [test] to install the testing dependencies, you can just drop into the hatch testing environment and the dependencies will be synced automatically.
