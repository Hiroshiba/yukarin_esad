"""Git情報取得ユーティリティ"""

import subprocess


def get_commit_id():
    """GitコミットIDを取得"""
    try:
        return (
            subprocess.check_output("git rev-parse HEAD", shell=True).decode().strip()
        )
    except Exception:
        return None


def get_branch_name():
    """Gitブランチ名を取得"""
    try:
        return (
            subprocess.check_output("git rev-parse --abbrev-ref HEAD", shell=True)
            .decode()
            .strip()
        )
    except Exception:
        return None
