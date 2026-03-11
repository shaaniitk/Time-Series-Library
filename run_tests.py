import os
import sys


def main():
    repo_dir = os.path.abspath(os.path.dirname(__file__))
    venv_python = os.path.join(repo_dir, 'ai_env', 'bin', 'python')

    if os.path.exists(venv_python) and os.path.realpath(sys.executable) != os.path.realpath(venv_python):
        os.execv(venv_python, [venv_python, __file__, *sys.argv[1:]])

    pytest_args = sys.argv[1:] if len(sys.argv) > 1 else ['tests']
    os.execv(sys.executable, [sys.executable, '-m', 'pytest', *pytest_args])


if __name__ == '__main__':
    main()
