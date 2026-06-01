"""
AST Analyzer Framework
======================
Provides abstract ASTAnalyzer interface and concrete implementations for different languages.
Used to generate function-to-test mappings.
"""

import os
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from utils import get_spec


class ASTAnalyzer(ABC):
    """Abstract base class for AST analyzers."""

    def __init__(self, project_name: str, project_path: str, output_dir: str):
        """
        Initialize AST analyzer.

        Args:
            project_name: Name of the project
            project_path: Absolute path to the project directory
            output_dir: Directory where to save the mapping file
        """
        self.project_name = project_name
        self.project_path = project_path
        self.output_dir = output_dir
        self.spec = get_spec(project_name)

        # Output file path
        self.output_file = os.path.join(output_dir, "function_testunit_mapping.json")

    @abstractmethod
    def generate_mapping(self) -> bool:
        """
        Generate function-to-test mapping file.

        Returns:
            bool: True if successful, False otherwise
        """
        pass

    def mapping_exists(self) -> bool:
        """Check if the mapping file already exists."""
        return os.path.exists(self.output_file)

    def get_output_path(self) -> str:
        """Get the path to the output mapping file."""
        return self.output_file

    @abstractmethod
    def get_language(self) -> str:
        """Get the language this analyzer handles."""
        pass


class PythonASTAnalyzer(ASTAnalyzer):
    """AST analyzer for Python projects using coverage + pytest."""

    def get_language(self) -> str:
        return "python"

    def generate_mapping(self) -> bool:
        """
        Generate function-to-test mapping for Python projects.

        Uses coverage.py with pytest to track which tests exercise which functions.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Import the generate_function_mapping from ast_analyze_python.py
            from ast_analyze_python import generate_function_mapping

            print(f"[PythonASTAnalyzer] Generating mapping for {self.project_name}...")
            print(f"[PythonASTAnalyzer] Project path: {self.project_path}")
            print(f"[PythonASTAnalyzer] Output: {self.output_file}")

            # Call the Python AST analyzer
            # Note: generate_function_mapping expects project_root to be the parent of project
            project_root = str(Path(self.project_path).parent)
            result = generate_function_mapping(
                project_name=self.project_name,
                project_path=project_root,
                output_file=self.output_file
            )

            if result == 0 and self.mapping_exists():
                print(f"[PythonASTAnalyzer] ✓ Mapping generated successfully")
                return True
            else:
                print(f"[PythonASTAnalyzer] ✗ Mapping generation failed")
                return False

        except Exception as e:
            print(f"[PythonASTAnalyzer] Error generating mapping: {e}")
            import traceback
            traceback.print_exc()
            return False


class JavaASTAnalyzer(ASTAnalyzer):
    """AST analyzer for Java projects using JaCoCo + Maven."""

    def get_language(self) -> str:
        return "java"

    def generate_mapping(self) -> bool:
        """
        Generate function-to-test mapping for Java projects.

        Uses JaCoCo with Maven to track which tests exercise which methods.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Import the generate_java_function_mapping from ast_analyze_java.py
            from ast_analyze_java import generate_java_function_mapping

            print(f"[JavaASTAnalyzer] Generating mapping for {self.project_name}...")
            print(f"[JavaASTAnalyzer] Project path: {self.project_path}")
            print(f"[JavaASTAnalyzer] Output: {self.output_file}")

            # Call the Java AST analyzer
            # Note: generate_java_function_mapping expects project_root to be the parent of project
            project_root = str(Path(self.project_path).parent)
            result = generate_java_function_mapping(
                project_name=self.project_name,
                project_path=project_root,
                output_dir=self.output_dir
            )

            if result == 0 and self.mapping_exists():
                print(f"[JavaASTAnalyzer] ✓ Mapping generated successfully")
                return True
            else:
                print(f"[JavaASTAnalyzer] ✗ Mapping generation failed")
                return False

        except Exception as e:
            print(f"[JavaASTAnalyzer] Error generating mapping: {e}")
            import traceback
            traceback.print_exc()
            return False


class GoASTAnalyzer(ASTAnalyzer):
    """AST analyzer for Go projects using go test coverage."""

    def get_language(self) -> str:
        return "go"

    def generate_mapping(self) -> bool:
        try:
            from ast_analyze_go import generate_go_function_mapping

            print(f"[GoASTAnalyzer] Generating mapping for {self.project_name}...")
            print(f"[GoASTAnalyzer] Project path: {self.project_path}")
            print(f"[GoASTAnalyzer] Output: {self.output_file}")

            project_root = str(Path(self.project_path).parent)
            result = generate_go_function_mapping(
                project_name=self.project_name,
                project_path=project_root,
                output_dir=self.output_dir,
            )

            if result == 0 and self.mapping_exists():
                print(f"[GoASTAnalyzer] ✓ Mapping generated successfully")
                return True
            else:
                print(f"[GoASTAnalyzer] ✗ Mapping generation failed")
                return False

        except Exception as e:
            print(f"[GoASTAnalyzer] Error generating mapping: {e}")
            import traceback
            traceback.print_exc()
            return False


class ASTAnalyzerFactory:
    """Factory for creating AST analyzers based on project language."""

    _analyzers = {
        "python": PythonASTAnalyzer,
        "java": JavaASTAnalyzer,
        "go": GoASTAnalyzer,
    }

    @classmethod
    def create_analyzer(
        cls,
        project_name: str,
        project_path: str,
        output_dir: str
    ) -> ASTAnalyzer:
        """
        Create an AST analyzer for the given project.

        Args:
            project_name: Name of the project
            project_path: Path to the project directory
            output_dir: Directory where to save the mapping file

        Returns:
            ASTAnalyzer instance

        Raises:
            ValueError: If project language is not supported
        """
        spec = get_spec(project_name)
        if spec is None:
            raise ValueError(f"No spec found for project: {project_name}")

        language = spec.get("language", "python").lower()

        analyzer_class = cls._analyzers.get(language)
        if analyzer_class is None:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported languages: {', '.join(cls._analyzers.keys())}"
            )

        return analyzer_class(project_name, project_path, output_dir)

    @classmethod
    def register_analyzer(cls, language: str, analyzer_class: type):
        """
        Register a custom AST analyzer for a language.

        Args:
            language: Language identifier (e.g., "javascript", "go")
            analyzer_class: ASTAnalyzer subclass
        """
        if not issubclass(analyzer_class, ASTAnalyzer):
            raise TypeError("analyzer_class must be a subclass of ASTAnalyzer")
        cls._analyzers[language.lower()] = analyzer_class

    @classmethod
    def get_supported_languages(cls) -> list:
        """Get list of supported languages."""
        return list(cls._analyzers.keys())


def ensure_mapping_exists(
    project_name: str,
    project_path: str,
    output_dir: str,
    force_regenerate: bool = False
) -> bool:
    """
    Ensure that function_testunit_mapping.json exists for the project.

    If the mapping file doesn't exist (or force_regenerate=True), generates it
    using the appropriate AST analyzer based on project language.

    Args:
        project_name: Name of the project
        project_path: Path to the project directory
        output_dir: Directory where to save/check for the mapping file
        force_regenerate: If True, regenerate even if file exists

    Returns:
        bool: True if mapping exists or was successfully generated, False otherwise
    """
    try:
        analyzer = ASTAnalyzerFactory.create_analyzer(
            project_name=project_name,
            project_path=project_path,
            output_dir=output_dir
        )

        # Check if mapping already exists
        if not force_regenerate and analyzer.mapping_exists():
            print(f"[{analyzer.get_language().upper()}] Mapping file already exists: {analyzer.get_output_path()}")
            return True

        # Generate mapping
        print(f"[{analyzer.get_language().upper()}] Mapping file not found, generating...")
        success = analyzer.generate_mapping()

        if success:
            print(f"[{analyzer.get_language().upper()}] ✓ Mapping file created: {analyzer.get_output_path()}")
        else:
            print(f"[{analyzer.get_language().upper()}] ✗ Failed to create mapping file")

        return success

    except ValueError as e:
        print(f"[ERROR] Cannot create analyzer: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
