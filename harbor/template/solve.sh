
WORKDIR="/workspace/project/{repo_name}"
PATCH_FILE="/workspace/gt.diff"

git -C "$WORKDIR" apply "$PATCH_FILE"