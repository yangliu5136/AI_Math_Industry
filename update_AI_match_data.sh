#!/bin/bash

# 要删除的chroma_db  storage目录
DB_DIR="chroma_db"

# 检查目录是否存在
if [ -d "$DB_DIR" ]; then
    echo "正在清理 $DB_DIR 目录..."

    # 删除目录
    rm -rf "${DB_DIR}"

    # 检查是否删除成功
    if [ ! -d "$DB_DIR" ]; then
        echo "$DB_DIR 目录已清空"
    else
        echo "错误：清理 $DB_DIR 目录时出现问题" >&2
        exit 1
    fi
else
    echo "警告：目录 $DB_DIR 不存在" >&2
    exit 1
fi

STORAGE_DIR="storage"
# 检查目录是否存在
if [ -d "$STORAGE_DIR" ]; then
    echo "正在清理 $STORAGE_DIR 目录..."

    # 删除目录
    rm -rf "${STORAGE_DIR}"

    # 检查是否删除成功
    if [ ! -d "$STORAGE_DIR" ]; then
        echo "$STORAGE_DIR 目录已清空"
    else
        echo "错误：清理 $STORAGE_DIR 目录时出现问题" >&2
        exit 1
    fi
else
    echo "警告：目录 $STORAGE_DIR 不存在" >&2
    exit 1
fi

# 删除redis缓存
AI_MATCH_REDIS="redis_control/deleteMatchRedisData.py"
# 检查py文件是否存在
if [ -f "$AI_MATCH_REDIS" ]; then
    python "$AI_MATCH_REDIS"
    echo "删除redis缓存成功"
fi
# 重新拉取数据
#if [ -f "getData.py" ]; then
#    python "getData.py"
#    echo "获取接口数据成功"
#fi
## 重新生成csv数据
#if [ -f "processData.py" ]; then
#    python "processData.py"
#    echo "重新生成csv数据成功"
#fi