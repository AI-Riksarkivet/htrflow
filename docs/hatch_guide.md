# Quick tutorial on Hatch usage

## Configuration
Hatch can be configured in [tool.hatch] blocks in pyproject.toml or in hatch.toml without the tool header. I've created a hatch.toml configuration as it allows more clarity in what is hatch configuration and what isn't and makes moving to other build systems if necessary much easier.

## Environments
Hatch has the ability to manage environments for different purposes with different packages. I've set up a test environment, which means that instead of having to install with [test] to install the testing dependencies, you can just drop into the hatch testing environment and the dependencies will be synced automatically.

Commands can be run in an environment with ```hatch env run <environment:command>```. For instance we can run pytest in the testing environment by running ```hatch env run test:pytest```. You can also drop directly into an environment's shell with ```hatch shell <environment>```. This will sync dependencies and get you all set up in that environment; for instance dropping into the test environment will give you access to all the testing dependencies automatically.

## Matrices
This is identical to Github workflow matrices, but defined in the project system instead. I've set up some examples basic matrices that we can fill out further going forward.

## Dependencies
Dependencies are managed as usual, however it's worth noting that by associating optional dependencies (such as for testing and doc-building) with environments, we can use those dependencies by dropping into the appropriate environment instead of having to install with optional dependencies.
