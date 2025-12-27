#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
update_asset.py

- 각 user의 마지막 자산 로그(date)를 기준으로
- date + 1 의 자산을 계산하여 user_asset_log에 INSERT
- 로그가 없는 user는 user_info를 seed로 사용
- 환율(eiExchangeRate.usdkrw)이 NULL이면 자산 유지
"""

from __future__ import annotations

from pathlib import Path
from datetime import timedelta, date
import os
import pymysql
from dotenv import load_dotenv
from decimal import Decimal


# ---------------------------------------------------------------------------
# .env 로드
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
else:
    load_dotenv(override=True)


def get_connection() -> pymysql.Connection:
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )


# ---------------------------------------------------------------------------
# 메인 로직
# ---------------------------------------------------------------------------
def update_assets() -> None:
    conn = get_connection()

    try:
        with conn.cursor() as cursor:

            # ------------------------------------------------------------------
            # 1️⃣ 마지막 로그가 있는 user
            # ------------------------------------------------------------------
            cursor.execute(
                """
                SELECT
                    l.user_id,
                    l.date,
                    l.hedged_etf,
                    l.unhedged_etf
                FROM user_asset_log l
                JOIN (
                    SELECT user_id, MAX(date) AS max_date
                    FROM user_asset_log
                    GROUP BY user_id
                ) t
                  ON l.user_id = t.user_id
                 AND l.date = t.max_date
                """
            )
            latest_logs = cursor.fetchall()

            latest_map = {row["user_id"]: row for row in latest_logs}

            # ------------------------------------------------------------------
            # 2️⃣ 전체 user_info
            # ------------------------------------------------------------------
            cursor.execute(
                """
                SELECT
                    user_id,
                    date,
                    hedged_etf,
                    unhedged_etf
                FROM user_info
                """
            )
            users = cursor.fetchall()

            if not users:
                print("[WARN] user_info 테이블이 비어 있습니다.")
                return

            # ------------------------------------------------------------------
            # 3️⃣ 기준 날짜 결정: 로그 테이블의 max date 또는 user_info의 min date
            # ------------------------------------------------------------------
            cursor.execute(
                """
                SELECT MAX(date) as max_date
                FROM user_asset_log
                """
            )
            max_log_date_result = cursor.fetchone()
            max_log_date = max_log_date_result["max_date"] if max_log_date_result and max_log_date_result["max_date"] else None
            
            if max_log_date:
                # 로그 테이블에 데이터가 있으면 최신 날짜 사용
                reference_date = max_log_date
            else:
                # 로그 테이블이 비어있으면 user_info의 가장 빠른 날짜 사용
                if users:
                    min_user_info_date = min(u["date"] for u in users if u["date"])
                    reference_date = min_user_info_date
                    print(f"📅 로그 테이블이 비어있어 user_info의 최소 날짜 사용: {reference_date}")
                else:
                    print("[WARN] user_info 테이블이 비어 있습니다.")
                    return

            # ------------------------------------------------------------------
            # 4️⃣ user_info를 date 순서로 정렬 (빠른 날짜부터 처리)
            # ------------------------------------------------------------------
            users_sorted = sorted(users, key=lambda u: u["date"] if u["date"] else date.max)
            
            # 기준 날짜 이하인 user_info.date를 가진 유저만 필터링
            users_to_process = [u for u in users_sorted if u["date"] and u["date"] <= reference_date]

            # ------------------------------------------------------------------
            # 5️⃣ user별 자산 계산
            # ------------------------------------------------------------------
            for user in users_to_process:
                user_id = user["user_id"]
                user_info_date = user["date"]
                user_info_hedged = Decimal(str(user["hedged_etf"])) if user["hedged_etf"] is not None else Decimal("0")
                user_info_unhedged = Decimal(str(user["unhedged_etf"])) if user["unhedged_etf"] is not None else Decimal("0")

                # user_info의 date가 user_asset_log에 있는지 확인
                cursor.execute(
                    """
                    SELECT COUNT(*) as cnt
                    FROM user_asset_log
                    WHERE user_id = %s AND date = %s
                    """,
                    (user_id, user_info_date),
                )
                has_user_info_date = cursor.fetchone()["cnt"] > 0

                # user_info의 date가 없으면 먼저 INSERT (hedged_etf, unhedged_etf 그대로)
                if not has_user_info_date:
                    cursor.execute(
                        """
                        SELECT usdkrw
                        FROM eiExchangeRate
                        WHERE date = %s
                        """,
                        (user_info_date,),
                    )
                    fx_at_date = cursor.fetchone()
                    fx_rate = (
                        Decimal(str(fx_at_date["usdkrw"]))
                        if fx_at_date and fx_at_date["usdkrw"] is not None
                        else None
                    )

                    cursor.execute(
                        """
                        INSERT INTO user_asset_log (
                            user_id,
                            date,
                            hedged_etf,
                            unhedged_etf,
                            fx_rate,
                            fx_return
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            user_id,
                            user_info_date,
                            float(user_info_hedged),
                            float(user_info_unhedged),
                            float(fx_rate) if fx_rate is not None else None,
                            None,  # user_info의 date는 fx_return이 없음
                        ),
                    )
                    print(f"✅ {user_id} | {user_info_date} 로그 추가")

                # 기준값 결정 (다음날부터 계산하기 위한 기준)
                if user_id in latest_map:
                    base = latest_map[user_id]
                    base_date = base["date"]
                    hedged_etf = Decimal(str(base["hedged_etf"]))
                    unhedged_etf = Decimal(str(base["unhedged_etf"]))
                else:
                    # user_info의 date를 기준으로 다음날부터 계산
                    base_date = user_info_date
                    hedged_etf = user_info_hedged
                    unhedged_etf = user_info_unhedged

                next_date = base_date + timedelta(days=1)

                # ------------------------------------------------------------------
                # 6️⃣ 환율 조회
                # ------------------------------------------------------------------
                cursor.execute(
                    """
                    SELECT usdkrw
                    FROM eiExchangeRate
                    WHERE date = %s
                    """,
                    (base_date,),
                )
                prev_fx = cursor.fetchone()

                cursor.execute(
                    """
                    SELECT usdkrw
                    FROM eiExchangeRate
                    WHERE date = %s
                    """,
                    (next_date,),
                )
                next_fx = cursor.fetchone()

                prev_rate = (
                    Decimal(str(prev_fx["usdkrw"]))
                    if prev_fx and prev_fx["usdkrw"] is not None
                    else None
                )
                next_rate = (
                    Decimal(str(next_fx["usdkrw"]))
                    if next_fx and next_fx["usdkrw"] is not None
                    else None
                )

                # ------------------------------------------------------------------
                # 7️⃣ 환율 반영
                # ------------------------------------------------------------------
                if prev_rate is None or next_rate is None:
                    fx_return = None
                    new_unhedged = unhedged_etf
                else:
                    fx_return = (next_rate / prev_rate) - Decimal("1")
                    new_unhedged = unhedged_etf * (Decimal("1") + fx_return)

                # ------------------------------------------------------------------
                # 8️⃣ 로그 INSERT
                # ------------------------------------------------------------------
                cursor.execute(
                    """
                    INSERT INTO user_asset_log (
                        user_id,
                        date,
                        hedged_etf,
                        unhedged_etf,
                        fx_rate,
                        fx_return
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        user_id,
                        next_date,
                        float(hedged_etf),
                        float(new_unhedged),
                        float(next_rate) if next_rate is not None else None,
                        float(fx_return) if fx_return is not None else None,
                    ),
                )

                print(f"✅ {user_id} | {next_date} 로그 추가")

            conn.commit()
            print("🎉 모든 user 자산 로그 INSERT 완료")

    except Exception as e:
        conn.rollback()
        print(f"❌ 자산 업데이트 실패: {e}")
        raise
    finally:
        conn.close()


def main() -> None:
    update_assets()


if __name__ == "__main__":
    main()
