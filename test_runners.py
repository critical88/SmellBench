"""
Test Runner Framework
=====================
Provides abstract TestRunner interface and concrete implementations for different languages.
"""

import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils import pushd, _log, create_test_command, DEBUG_LOG_LEVEL, conda_exec_cmd, get_spec


class TestRunner(ABC):
    """Abstract base class for test runners."""

    def __init__(self, project_name: str, project_path: str):
        """
        Initialize test runner.

        Args:
            project_name: Name of the project
            project_path: Absolute path to the project directory
        """
        self.project_name = project_name
        self.project_path = project_path
        self.spec = get_spec(project_name)

    @abstractmethod
    def run_tests(
        self,
        test_file_paths: List[str],
        envs: Optional[Dict[str, str]] = None,
        test_cmd: str = "",
        timeout: Optional[int] = None,
        capture_output: bool = True,
    ) -> Tuple[bool, str]:
        """
        Run tests for the project.

        Args:
            test_file_paths: List of test file paths or test identifiers
            envs: Environment variables to set
            test_cmd: Additional test command arguments
            timeout: Timeout in seconds
            capture_output: If True, capture output; if False, stream to console

        Returns:
            (success: bool, output: str) - Test result and output
        """
        pass

    @abstractmethod
    def get_default_timeout(self) -> int:
        """Get the default timeout for this test runner."""
        pass

    def _prepare_env(self, envs: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Prepare environment variables for test execution."""
        exec_env = os.environ.copy()
        if envs:
            exec_env.update(envs)
        return exec_env


class PythonTestRunner(TestRunner):
    """Test runner for Python projects using pytest."""

    def get_default_timeout(self) -> int:
        """Python tests typically run faster."""
        return 60

    def run_tests(
        self,
        test_file_paths: List[str],
        envs: Optional[Dict[str, str]] = None,
        test_cmd: str = "",
        timeout: Optional[int] = None,
        capture_output: bool = True,
    ) -> Tuple[bool, str]:
        """
        Run Python tests via pytest in conda environment.

        Args:
            test_file_paths: List of test file paths (e.g., ["tests/test_core.py"])
            envs: Environment variables
            test_cmd: Additional pytest arguments
            timeout: Timeout in seconds
            capture_output: If True, capture output; if False, stream to console

        Returns:
            (success, output)
        """
        if timeout is None:
            timeout = self.get_default_timeout()

        try:
            with pushd(self.project_path):
                exec_env = self._prepare_env(envs)

                # Build pytest command
                cmd = create_test_command(test_file_paths, test_cmd=test_cmd)
                _log(f"Running Python tests: {' '.join(cmd)}", DEBUG_LOG_LEVEL)

                # Execute via conda
                result = conda_exec_cmd(
                    cmd,
                    spec=self.spec,
                    cwd=".",
                    envs=exec_env,
                    capture_output=capture_output,
                    timeout=timeout,
                )

                success = result.returncode == 0
                output = result.stdout if result.stdout else ""

                if not success:
                    _log(f"Python tests failed: {output}", DEBUG_LOG_LEVEL)

                return success, output

        except subprocess.TimeoutExpired:
            error_msg = f"Python test execution timed out after {timeout}s"
            print(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Error running Python tests: {e}"
            print(error_msg)
            return False, error_msg


class JavaTestRunner(TestRunner):
    """Test runner for Java projects using Maven."""

    def get_default_timeout(self) -> int:
        """Java/Maven tests typically need more time."""
        return 120

    def _build_maven_test_cmd(
        self,
        test_file_paths: List[str],
        test_cmd: str = "",
    ) -> List[str]:
        """
        Build a Maven test command.

        Args:
            test_file_paths: Test identifiers, can be:
                - Test class simple names: ["FileUtilsTest", "IOCaseTest"]
                - Test class FQNs: ["org.apache.commons.io.FileUtilsTest"]
                - Test class + method: ["FileUtilsTest#testCopy"]
            test_cmd: Additional Maven arguments (may include -pl for multi-module projects)

        Returns:
            Maven command as list of strings
        """
        cmd = ["mvn", "test"]

        if test_file_paths:
            test_pattern = ",".join(test_file_paths)
            cmd.append(f"-Dtest={test_pattern}")

        # Maven flags
        cmd.append("-DfailIfNoTests=false")
        cmd.append("-Dmaven.test.failure.ignore=false")

        # Only add default -pl if test_cmd doesn't already specify it
        # This avoids duplicate -pl flags for multi-module projects
        if test_cmd and "-pl" in test_cmd:
            # test_cmd already has -pl specification (e.g., "-pl gson")
            # Don't add our own -pl
            pass
        else:
            # Single-module project or no -pl specified, use current directory
            cmd.append("-pl")
            cmd.append(".")

        # Additional arguments (may include -pl for multi-module projects)
        if test_cmd:
            cmd.extend(test_cmd.split())

        return cmd

    def run_tests(
        self,
        test_file_paths: List[str],
        envs: Optional[Dict[str, str]] = None,
        test_cmd: str = "",
        timeout: Optional[int] = None,
        capture_output: bool = True,
    ) -> Tuple[bool, str]:
        """
        Run Java tests via Maven.

        Args:
            test_file_paths: List of test identifiers
            envs: Environment variables
            test_cmd: Additional Maven arguments
            timeout: Timeout in seconds
            capture_output: If True, capture output; if False, stream to console

        Returns:
            (success, output)
        """
        if timeout is None:
            timeout = self.get_default_timeout()

        try:
            with pushd(self.project_path):
                exec_env = self._prepare_env(envs)

                # Build Maven command
                cmd = self._build_maven_test_cmd(test_file_paths, test_cmd=test_cmd)
                _log(f"Running Java tests: {' '.join(cmd)}", DEBUG_LOG_LEVEL)

                # Execute Maven
                result = subprocess.run(
                    cmd,
                    cwd=".",
                    capture_output=capture_output,
                    text=True,
                    env=exec_env,
                    timeout=timeout,
                )

                success = result.returncode == 0
                if capture_output:
                    output = result.stdout + "\n" + result.stderr
                else:
                    output = "(output streamed to console)"

                if not success and capture_output:
                    _log(f"Java tests failed: {output}", DEBUG_LOG_LEVEL)

                return success, output

        except subprocess.TimeoutExpired:
            error_msg = f"Java test execution timed out after {timeout}s"
            print(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Error running Java tests: {e}"
            print(error_msg)
            return False, error_msg


class TestRunnerFactory:
    """Factory for creating test runners based on project language."""

    _runners = {
        "python": PythonTestRunner,
        "java": JavaTestRunner,
    }

    @classmethod
    def create_runner(cls, project_name: str, project_path: str) -> TestRunner:
        """
        Create a test runner for the given project.

        Args:
            project_name: Name of the project
            project_path: Path to the project directory

        Returns:
            TestRunner instance

        Raises:
            ValueError: If project language is not supported
        """
        spec = get_spec(project_name)
        if spec is None:
            raise ValueError(f"No spec found for project: {project_name}")

        language = spec.get("language", "python").lower()

        runner_class = cls._runners.get(language)
        if runner_class is None:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported languages: {', '.join(cls._runners.keys())}"
            )

        return runner_class(project_name, project_path)

    @classmethod
    def register_runner(cls, language: str, runner_class: type):
        """
        Register a custom test runner for a language.

        Args:
            language: Language identifier (e.g., "javascript", "go")
            runner_class: TestRunner subclass
        """
        if not issubclass(runner_class, TestRunner):
            raise TypeError("runner_class must be a subclass of TestRunner")
        cls._runners[language.lower()] = runner_class

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Get list of supported languages."""
        return list(cls._runners.keys())
