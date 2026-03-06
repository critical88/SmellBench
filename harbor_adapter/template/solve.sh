
cat > /workspace/solution_patch.diff << '__SOLUTION__'
{patch}
__SOLUTION__

cd /workspace/project/{repo_name}

patch --fuzz=5 -p1 -i /workspace/solution_patch.diff