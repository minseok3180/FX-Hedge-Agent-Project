#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ECOS → 전처리/머지 → MariaDB 적재까지 한 번에 수행하는 스크립트.

기존 ecos_client.py, ecos_utils.py, ecos_main.py의 기능을 하나로 묶은 버전이다.
- ECOS API 호출
- 시계열 전처리 및 일단위 머지
- CSV 저장
- MariaDB(eiExchangeRate) 적재
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import pymysql
from dotenv import load_dotenv

from src.utils.settings import settings
from ecos_client import fetch_auto
from ecos_utils import (
    SeriesSpec,
    load_specs,
    to_value_frame,
    normalize_date_by_cycle,
    expand_to_daily,
    merge_wide,
)

# ---------------------------------------------------------------------------
# 환경 설정 (.env 로드)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
else:
    load_dotenv(override=True)


# ---------------------------------------------------------------------------
# ECOS 수집 및 병합
# ---------------------------------------------------------------------------

def fetch_one_series(
    spec: SeriesSpec,
    start: str,
    end: str,
) -> Tuple[str, pd.DataFrame]:
    """
    단일 SeriesSpec에 대해 ECOS에서 시계열을 가져와
    ['date', 'value'] → 일단위 확장까지 마친 DataFrame을 반환한다.
    """
    stat = spec.stat_code
    cyc = spec.cycle.upper()
    name = spec.name if spec.name else f"value_{stat}"

    print(f"[ECOS] {stat} ({cyc}) {start}~{end} 수집 중...")

    # 1) 원자료 수집
    raw = fetch_auto(
        stat_code=stat,
        cycle=cyc,
        item_code1=spec.item_code1,
        item_code2=spec.item_code2,
        start=start,
        end=end,
    )

    # 2) ['date', 'value']로 축소
    slim = to_value_frame(raw, cyc)

    # 3) 월/분기/연 값을 기간 시작일로 정규화
    slim = normalize_date_by_cycle(slim, cyc)

    # 4) 일단위로 확장
    daily = expand_to_daily(slim, cyc, start, end)

    return name, daily


def build_ecos_wide(
    start: str,
    end: str,
    specs_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    series_specs.csv를 기반으로 모든 ECOS 시계열을 수집하여
    날짜 기준 와이드 형태의 DataFrame으로 반환한다.
    """
    # 사양 파일 경로 결정
    if specs_path is None:
        specs_path = PROJECT_ROOT / "data" / "series_specs.csv"
    else:
        specs_path = Path(specs_path)
        if not specs_path.is_absolute():
            specs_path = PROJECT_ROOT / specs_path

    if not specs_path.exists():
        raise FileNotFoundError(f"사양 파일을 찾을 수 없습니다: {specs_path}")

    specs = load_specs(str(specs_path))

    series_wide_inputs: List[Tuple[str, pd.DataFrame]] = []

    for i, sp in enumerate(specs, 1):
        name, daily = fetch_one_series(sp, start, end)
        series_wide_inputs.append((name, daily))

    # 와이드 병합
    wide = merge_wide(series_wide_inputs)

    # 전역 기간으로 다시 인덱싱 후 ffill
    wide = wide.set_index("date")
    wide = wide.sort_index()

    full_index = pd.date_range(start=start, end=end, freq="D")
    wide = wide.reindex(full_index)
    wide.index.name = "date"
    wide = wide.ffill().reset_index()

    return wide


# ---------------------------------------------------------------------------
# MariaDB 적재
# ---------------------------------------------------------------------------

def upload_ecos_to_db(
    df: pd.DataFrame,
    table_name: str = "eiExchangeRate",
) -> None:
    """
    ECOS 머지 결과 DataFrame을 MariaDB에 업로드한다.
    기본 동작은 기존 테이블 데이터를 TRUNCATE 후 전체 재적재이다.
    """
    # 컬럼명 정리
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "date" not in df.columns:
        raise ValueError("DataFrame에 'date' 컬럼이 없습니다.")

    # DB 연결 정보는 settings를 사용
    print("데이터베이스 연결 중...")
    conn = pymysql.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        database=settings.db_name,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.Cursor,
        autocommit=False,
    )
    cursor = conn.cursor()

    try:
        # 테이블 존재 여부 확인
        cursor.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name = %s",
            (settings.db_name, table_name),
        )
        table_exists = cursor.fetchone()[0] > 0

        if not table_exists:
            print(f"테이블이 존재하지 않습니다. 새로 생성합니다: {table_name}")

            # 컬럼 정의 생성
            column_definitions = ["`date` DATE NOT NULL PRIMARY KEY"]
            for col in df.columns:
                if col == "date":
                    continue
                sample_value = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if sample_value is not None and isinstance(sample_value, (int, float, np.number)):
                    col_type = "DOUBLE"
                else:
                    col_type = "TEXT"
                column_definitions.append(f"`{col}` {col_type}")

            create_sql = f"""
            CREATE TABLE `{table_name}` (
                {', '.join(column_definitions)}
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            cursor.execute(create_sql)
            conn.commit()
            print("테이블 생성 완료")
            table_columns = df.columns.tolist()
            valid_columns = df.columns.tolist()
        else:
            print(f"테이블이 이미 존재합니다. 기존 데이터를 대체합니다: {table_name}")
            cursor.execute(f"DESCRIBE `{table_name}`")
            table_columns = [row[0] for row in cursor.fetchall()]

            valid_columns = ["date"] + [
                col for col in df.columns
                if col != "date" and col in table_columns
            ]
            missing_columns = [
                col for col in df.columns
                if col != "date" and col not in table_columns
            ]

            if missing_columns:
                print(f"경고: 다음 컬럼들이 테이블에 없어 제외됩니다: {missing_columns}")

            print("기존 데이터 삭제 중(TRUNCATE)...")
            cursor.execute(f"TRUNCATE TABLE `{table_name}`")
            print("기존 데이터 삭제 완료")

        df_to_insert = df[valid_columns].copy()

        if len(valid_columns) <= 1:
            print("업로드할 컬럼이 없습니다. (date만 있음)")
            return

        placeholders = ",".join(["%s"] * len(valid_columns))
        columns_str = ",".join([f"`{col}`" for col in valid_columns])
        sql = f"INSERT INTO `{table_name}` ({columns_str}) VALUES ({placeholders})"

        print(f"업로드할 컬럼: {valid_columns}")
        print(f"데이터 업로드 중... (총 {len(df_to_insert)}행)")

        for idx, (_, row) in enumerate(df_to_insert.iterrows(), 1):
            values = tuple(row.replace({np.nan: None}))
            cursor.execute(sql, values)
            if idx % 100 == 0:
                print(f"  진행: {idx}/{len(df_to_insert)}")

        conn.commit()
        print(f"데이터베이스 업로드 완료 (총 {len(df_to_insert)}행)")

    finally:
        cursor.close()
        conn.close()


# ---------------------------------------------------------------------------
# 파이프라인 실행 함수 및 CLI
# ---------------------------------------------------------------------------

def run_ecos_pipeline(
    start: str,
    end: str,
    specs: Optional[str] = None,
    out: Optional[str] = None,
    upload_db: bool = False,
    table_name: str = "eiExchangeRate",
) -> pd.DataFrame:
    """
    ECOS → 머지 → CSV 저장 → MariaDB 업로드까지 한 번에 수행하는 함수.
    CLI에서 직접 호출하거나, 다른 모듈에서 import해서 사용할 수 있다.
    """
    # 1) ECOS 수집 및 머지
    specs_path = Path(specs) if specs is not None else None
    wide = build_ecos_wide(start=start, end=end, specs_path=specs_path)

    # 2) CSV 저장
    if out is None:
        output_path = PROJECT_ROOT / "data" / "ecos.csv"
    else:
        output_path = Path(out)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wide_to_save = wide.copy()
    wide_to_save["date"] = wide_to_save["date"].dt.strftime("%Y-%m-%d")
    wide_to_save.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"CSV 저장 완료: {output_path}")

    # 3) DB 업로드
    if upload_db:
        print("\n" + "=" * 50)
        print("MariaDB 업로드 시작")
        print("=" * 50)
        upload_ecos_to_db(wide, table_name=table_name)

    return wide


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", default=None, help="시계열 사양 CSV 경로 (기본값: data/series_specs.csv)")
    parser.add_argument("--start", required=True, help="전역 시작일 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="전역 종료일 YYYY-MM-DD")
    parser.add_argument("--out", default=None, help="최종 저장 경로(CSV) (기본값: data/ecos.csv)")
    parser.add_argument("--db", action="store_true", help="CSV 저장 후 MariaDB에 업로드")
    parser.add_argument("--table", default="eiExchangeRate", help="적재할 테이블명 (기본값: eiExchangeRate)")

    args = parser.parse_args()

    run_ecos_pipeline(
        start=args.start,
        end=args.end,
        specs=args.specs,
        out=args.out,
        upload_db=bool(args.db),
        table_name=args.table,
    )


if __name__ == "__main__":
    main()
