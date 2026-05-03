#!/usr/bin/env python
"""Test diff_analyzer module with sample diffs."""
import sys
from diff_analyzer import extract_modified_functions, print_modified_functions

# Sample Python diff
PYTHON_DIFF = """
diff --git a/click/core.py b/click/core.py
index 1234567..abcdefg 100644
--- a/click/core.py
+++ b/click/core.py
@@ -100,6 +100,7 @@ class Context:
     def __init__(self, command):
         self.command = command
+        self.invoked_subcommand = None

     def exit(self, code=0):
         sys.exit(code)
@@ -200,6 +201,10 @@ class Command:
         ctx = Context(self)
         return ctx

+    def invoke_subcommand(self, name):
+        # New method to invoke subcommands
+        return self.commands[name]
+
 def format_help(ctx):
     return ctx.get_help()
"""

# Sample Java diff
JAVA_DIFF = """
diff --git a/src/main/java/com/google/gson/Gson.java b/src/main/java/com/google/gson/Gson.java
index 1234567..abcdefg 100644
--- a/src/main/java/com/google/gson/Gson.java
+++ b/src/main/java/com/google/gson/Gson.java
@@ -500,6 +500,7 @@ public final class Gson {
   public <T> T fromJson(String json, Class<T> classOfT) {
     Object object = fromJson(json, (Type) classOfT);
+    // Added null check
     return Primitives.wrap(classOfT).cast(object);
   }

@@ -600,6 +601,11 @@ public final class Gson {
     return new GsonBuilder();
   }

+  public String toJsonTree(Object src) {
+    // New convenience method
+    return toJson(toJsonTree(src));
+  }
+
   private static class FutureTypeAdapter<T> extends TypeAdapter<T> {
     private TypeAdapter<T> delegate;
"""


def test_python():
    print("=" * 60)
    print("Testing Python diff analysis")
    print("=" * 60)

    # Create a temporary project structure
    import tempfile
    import os
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create click/core.py with sample code
        click_dir = Path(tmpdir) / "click"
        click_dir.mkdir()

        core_py = click_dir / "core.py"
        # Create a file with enough lines to match the diff line numbers
        padding_lines = ["# padding line"] * 95
        core_py.write_text("\n".join(padding_lines) + """
import sys

# Line ~98

class Context:
    def __init__(self, command):
        self.command = command
        self.invoked_subcommand = None

    def exit(self, code=0):
        sys.exit(code)
""" + "\n".join(["# more padding"] * 90) + """

# Line ~197

class Command:
    def __init__(self):
        self.commands = {}

    def make_context(self):
        ctx = Context(self)
        return ctx

    def invoke_subcommand(self, name):
        # New method to invoke subcommands
        return self.commands[name]

def format_help(ctx):
    return ctx.get_help()
""")

        # Extract functions from diff
        print(f"\nRepository path: {tmpdir}")
        print(f"Diff content length: {len(PYTHON_DIFF)} chars")
        functions = extract_modified_functions(
            diff_content=PYTHON_DIFF,
            repo_path=tmpdir,
            language="python",
            verbose=True,
        )

        print_modified_functions(functions)

        # Should find at least Context.__init__ and Command.invoke_subcommand
        # (may find more due to hunk range overlap)
        print(f"\nFound {len(functions)} modified functions")

        return len(functions) >= 2


def test_java():
    print("\n" + "=" * 60)
    print("Testing Java diff analysis")
    print("=" * 60)

    try:
        import javalang
    except ImportError:
        print("  javalang not installed, skipping Java test")
        return True

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create Java source file
        java_dir = Path(tmpdir) / "src" / "main" / "java" / "com" / "google" / "gson"
        java_dir.mkdir(parents=True)

        gson_java = java_dir / "Gson.java"
        # Create file with padding to match diff line numbers
        padding_lines = ["// padding line"] * 495
        gson_java.write_text("\n".join(padding_lines) + """
package com.google.gson;

// Line ~497

public final class Gson {

  public <T> T fromJson(String json, Class<T> classOfT) {
    Object object = fromJson(json, (Type) classOfT);
    // Added null check
    return Primitives.wrap(classOfT).cast(object);
  }
""" + "\n".join(["  // more padding"] * 90) + """

  public static GsonBuilder newBuilder() {
    return new GsonBuilder();
  }

  public String toJsonTree(Object src) {
    // New convenience method
    return toJson(toJsonTree(src));
  }

  private static class FutureTypeAdapter<T> extends TypeAdapter<T> {
    private TypeAdapter<T> delegate;
  }
}
""")

        # Extract functions from diff
        print(f"\nRepository path: {tmpdir}")
        functions = extract_modified_functions(
            diff_content=JAVA_DIFF,
            repo_path=tmpdir,
            language="java",
            verbose=True,
        )

        print_modified_functions(functions)

        # Should find at least fromJson and toJsonTree
        print(f"\nFound {len(functions)} modified functions")

        return len(functions) >= 2


if __name__ == "__main__":
    success = True

    try:
        success = test_python() and success
    except Exception as e:
        print(f"Python test failed: {e}")
        import traceback
        traceback.print_exc()
        success = False

    try:
        success = test_java() and success
    except Exception as e:
        print(f"Java test failed: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
