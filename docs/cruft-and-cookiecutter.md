# Cruft and Cookiecutter Usage

Here is a quick run through using Cruft and Cookiecutter for the sake of project structure. It's pretty simple, all things considered.

## Cruft and Cookiecutter
Cookiecutter is a way of managing project templates across multiple repos, with centralized config and info. Cruft allows us to automate a lot of the tedious parts of managing our local repos that are connected to these templates, but automatically updating our local projects when the template changes, without us manually having to move things around.

## Cookiecutter Templates
A cookiecutter template is stored as a normal github repo with some special formatting and configuration. I won't cover that in-depth here, but for a quick runthrough: the cookiecutter.json configures options and defaults that will be given when a new project is generated from the template. The folder name in double curly braces ("{{}}") is the project folder, everything outside that is information about the template which won't be cloned when a new project is generated. 

When a new project is generated, the curly braces folder name will be dynamically replaced with the project name and all of the contents will be cloned into your new project accordingly.

You can edit the template by just cloning the repo and pushing changes to github, as usual. Beyond this you shouldn't need to work with cookiecutter directly, as Cruft wraps the command line functionality.

## Project Management With Cruft

### Creating a project
A project can be created from a cookiecutter template by installing Cruft through typical Python channels, then running `cruft create <url to template>`. Alternatively, cloning or pulling from a repo running no cruft will already have everything set up to continue working with cruft without any further setup on your end. Cruft config is stored in a local .cruft.json file. From here you can proceed with development normally, the only concerns regarding Cruft are when the template changes.

### Updating the template
If the cookiecutter template changes, having things moved, renamed, etc., we can use Cruft to automatically update our local project to fit the new template.

First off, run `cruft check` in your project directory to check if an update is needed. From there, `cruft diff` will show the diff between your project and the update template. `cruft update` will pull the changes from the template's repo, offer to show you the diff, and then allow you to apply the changes.

**NOTE:** If there are merge conflicts when you attempt to update, cruft will throw a kind of ugly error and drop a *.rej file in the folder with the conflict, showing the diffs. You will have to resolve the merge conflict manually, or using git tools, before you can fully update. However, it does update in chunks so you will likely be able to get some updates even with a merge conflict.

**Note 2:** Cruft update will only run if your git working tree is clean, otherwise it'll tell you to clean up and exit. Make sure your tree is clean before updating.

That's about all there is to it, however `cruft` can be run with no arguments to get a help page detailing its subcommand, and each subcommand can be run with `--help` for information on its various options and flags.

