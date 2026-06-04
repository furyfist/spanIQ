import subprocess
import sys


def main() -> None:
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "test" and args[1] == "run":
        test_args = args[2:]
        sys.exit(subprocess.call([sys.executable, "-m", "pytest", *test_args, "-v"]))
    else:
        print("Usage: spaniq test run <test_file_or_dir>")
        sys.exit(1)
