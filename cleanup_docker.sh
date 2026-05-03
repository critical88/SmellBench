#!/bin/bash

# 清理selected repos的docker容器和镜像

# 定义要清理的repos数组
REPOS=(
    "click"
    "numpy"
    "jinja"
    "pandas"
    "seaborn"
    "sphinx"
    "sympy"
    "scikit-learn"
    "xarray"
)

echo "==================================="
echo "Docker Cleanup Script"
echo "==================================="

# 列出所有要清理的repos
echo ""
echo "Repos to cleanup:"
for repo in "${REPOS[@]}"; do
    echo "  - $repo"
done

echo ""
read -p "Do you want to proceed with cleanup? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 0
fi

# 删除每个repo的docker容器和镜像
for repo in "${REPOS[@]}"; do
  echo "===================="
  echo "Processing: $repo"

  # 删除容器
  echo "Removing containers for $repo..."
  containers=$(docker ps -a | grep "${repo}-" | awk '{print $1}')
  if [ -n "$containers" ]; then
    echo "$containers" | xargs docker rm -f
    echo "✓ Containers removed"
  else
    echo "- No containers found"
  fi

  # 删除镜像
  echo "Removing images for $repo..."
  images=$(docker images | grep "${repo}-" | awk '{print $3}')
  if [ -n "$images" ]; then
    echo "$images" | xargs docker rmi -f
    echo "✓ Images removed"
  else
    echo "- No images found"
  fi
done

echo "===================="
echo "✓ Cleanup completed!"
