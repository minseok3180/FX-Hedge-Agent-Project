#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
user_info 테이블에 새로운 열(risk_level)을 추가하고,
기존 3개의 행에 차례대로 0.2, 0.5, 0.8 값을 넣는 **순수 DB 스크립트**입니다.

- 에이전트/툴과 전혀 연결되지 않습니다.
- DB 접속 정보는 프로젝트 루트의 .env에서 직접 읽어옵니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import os

import pymysql
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# .env 로드 (DB_HOST, DB_USER, DB_PASSWORD, DB_NAME 등)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
else:
    load_dotenv(override=True)


def get_connection() -> pymysql.Connection:
    """환경 변수에서 DB 설정을 읽어 직접 MariaDB 커넥션을 생성한다."""
    host = os.getenv("DB_HOST")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    name = os.getenv("DB_NAME")
    port = int(os.getenv("DB_PORT", "3306"))

    if not all([host, user, password, name]):
        raise RuntimeError(
            "DB_HOST / DB_USER / DB_PASSWORD / DB_NAME 중 일부가 .env에 정의되지 않았습니다."
        )

    return pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=name,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )


def ensure_risk_level_column() -> None:
    """
    user_info 테이블에 risk_level 컬럼이 없으면 추가한다.
    이미 존재하면 에러를 무시하고 넘어간다.
    """
    conn = get_connection()

    alter_sql = """
        ALTER TABLE user_info
        ADD COLUMN risk_level DOUBLE DEFAULT NULL
    """

    try:
        with conn.cursor() as cursor:
            print("[INFO] user_info 테이블에 risk_level 컬럼 추가 시도")
            cursor.execute(alter_sql)
            conn.commit()
            print("✅ risk_level 컬럼 추가 완료")
    except Exception as e:
        # 이미 컬럼이 있는 경우 등은 경고만 남기고 계속 진행
        msg = str(e)
        if "Duplicate column name" in msg or "already exists" in msg:
            print(f"[WARN] risk_level 컬럼이 이미 존재합니다: {msg}")
        else:
            print(f"[ERROR] risk_level 컬럼 추가 중 오류 발생: {msg}")
            raise
    finally:
        conn.close()


def update_risk_levels(values: List[float]) -> None:
    """
    user_info 테이블의 앞 3개 행에 대해 risk_level 값을 순서대로 설정한다.

    Args:
        values: risk_level로 설정할 값 리스트 (예: [0.2, 0.5, 0.8])
    """
    if len(values) < 3:
        raise ValueError("values 리스트에는 최소 3개의 값이 있어야 합니다.")

    conn = get_connection()

    select_sql = """
        SELECT user_id
        FROM user_info
        ORDER BY user_id ASC
        LIMIT 3
    """
    update_sql = """
        UPDATE user_info
        SET risk_level = %s
        WHERE user_id = %s
    """

    try:
        with conn.cursor() as cursor:
            print("[INFO] user_info 테이블에서 상위 3개 user_id 조회")
            cursor.execute(select_sql)
            rows = cursor.fetchall()

            if not rows:
                print("[WARN] user_info 테이블에 행이 없습니다. 업데이트를 수행하지 않습니다.")
                return

            if len(rows) < 3:
                print(
                    f"[WARN] user_info 테이블에 행이 {len(rows)}개만 존재합니다. "
                    f"앞 {len(rows)}개 행에만 risk_level을 설정합니다."
                )

            for (row, risk) in zip(rows, values):
                user_id = row["user_id"]
                print(f"[INFO] user_id={user_id} 에 risk_level={risk} 설정")
                cursor.execute(update_sql, (risk, user_id))

            conn.commit()
            print("✅ risk_level 값 업데이트 완료")
    finally:
        conn.close()


def main() -> None:
    ensure_risk_level_column()
    # 현재 행이 3개라고 가정하고, 차례대로 0.2, 0.5, 0.8을 설정
    update_risk_levels([0.2, 0.5, 0.8])


if __name__ == "__main__":
    main()


