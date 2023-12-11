import subprocess

def main():
    print("Running 'git init'")
    subprocess.run(['git', 'init'], check=True)

    print("Running 'pip install --upgrade pip pre-commit poetry'")
    subprocess.run(['pip', 'install', '--upgrade', 'pip', 'pre-commit', 'poetry'], check=True)

    print("Running 'pre-commit install'")

    subprocess.run(['pre-commit', 'install'], check=True)

    print("Running 'poetry config --local virtualenvs.in-project true'")
    subprocess.run(['poetry', 'config', '--local', 'virtualenvs.in-project', 'true'], check=True)

    print("Running 'poetry install'")
    subprocess.run(['poetry', 'install'], check=True)

    print("Project successfully initialized")
    return

if __name__ == "__main__":
    main()
