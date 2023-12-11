import re
import sys

MODULE_REGEX = r"^[a-zA-Z][_a-zA-Z0-9]+$"
module_name = "{{ cookiecutter.repo_name }}"

if not re.match(MODULE_REGEX, module_name):
    print("ERROR: %s is not a valid Python module name!" % module_name)
    sys.exit(1)

SERVICE_REGEX = r"^[a-zA-Z][_a-zA-Z0-9]+$"
service_name = "{{ cookiecutter.service_name }}"

if not re.match(SERVICE_REGEX, service_name):
    print("ERROR: %s is not a valid serverless service name!" % service_name)
    sys.exit(1)
