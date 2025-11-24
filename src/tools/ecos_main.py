#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
사용 예:
python src/tools/ecos_main.py --start 2020-01-01 --end 2025-11-23 --db"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import pymysql
import numpy as np
from dotenv import load_dotenv

from ecos_client import fetch_auto
from ecos_utils import (
    load_specs,
    to_value_frame,
    expand_to_daily,
    normalize_date_by_cycle,
    merge_wide,
)

# .env 로드 (프로젝트 루트에서)
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
env_path = project_root / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    load_dotenv(override=True)

def upload_to_db(df: pd.DataFrame) -> None:
    """DataFrame을 MariaDB에 업로드 (한국어 컬럼명 그대로 사용)"""
    # DB 연결 정보
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = int(os.getenv("DB_PORT", 3306))
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
        raise ValueError("환경변수 DB_HOST, DB_NAME, DB_USER, DB_PASSWORD가 모두 설정되어야 합니다.")
    
    # 컬럼명 정리 (앞뒤 공백 제거)
    df.columns = df.columns.str.strip()
    
    # date 컬럼이 없으면 에러
    if "date" not in df.columns:
        raise ValueError("DataFrame에 'date' 컬럼이 없습니다.")
    
    # DB 연결
    print("데이터베이스 연결 중...")
    conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4"
    )
    cursor = conn.cursor()
    
    # 테이블 존재 여부 확인
    cursor.execute("""
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE table_schema = %s AND table_name = 'eiExchangeRate'
    """, (DB_NAME,))
    table_exists = cursor.fetchone()[0] > 0
    
    if table_exists:
        # 테이블이 존재하면 삭제하고 새로 생성 (컬럼 구조가 다를 수 있으므로)
        print("기존 테이블을 삭제하고 새로 생성합니다...")
        cursor.execute("DROP TABLE IF EXISTS eiExchangeRate")
        conn.commit()
        table_exists = False
    
    if not table_exists:
        # 테이블이 없으면 새로 생성
        print("테이블이 존재하지 않습니다. 새로 생성 중...")
        
        # 컬럼 정의 생성
        column_definitions = ["`date` DATE NOT NULL PRIMARY KEY"]
        
        for col in df.columns:
            if col == "date":
                continue
            # 데이터 타입 추정 (숫자형은 DOUBLE, 그 외는 TEXT)
            sample_value = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if sample_value is not None and isinstance(sample_value, (int, float)):
                col_type = "DOUBLE"
            else:
                col_type = "TEXT"
            
            column_definitions.append(f"`{col}` {col_type}")
        
        create_table_sql = f"""
        CREATE TABLE eiExchangeRate (
            {', '.join(column_definitions)}
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        
        cursor.execute(create_table_sql)
        conn.commit()
        print("✅ 테이블 생성 완료")
        table_columns = df.columns.tolist()
        valid_columns = df.columns.tolist()
    else:
        # 테이블이 존재하면 구조 확인
        print("테이블이 존재합니다. 기존 데이터를 대체합니다.")
        cursor.execute("DESCRIBE eiExchangeRate")
        table_columns = [row[0] for row in cursor.fetchall()]
        
        # date와 CSV의 컬럼 중 테이블에 존재하는 것만 선택 (한국어 컬럼명 그대로 사용)
        valid_columns = ["date"] + [col for col in df.columns if col != "date" and col in table_columns]
        missing_columns = [col for col in df.columns if col != "date" and col not in table_columns]
        
        if missing_columns:
            print(f"⚠️  경고: 다음 컬럼들이 테이블에 없어 제외됩니다: {missing_columns}")
        
        # 기존 데이터 삭제
        print("기존 데이터 삭제 중...")
        cursor.execute("TRUNCATE TABLE eiExchangeRate")
        print("✅ 기존 데이터 삭제 완료")
    
    # 사용할 컬럼만 선택
    df_to_insert = df[valid_columns].copy()
    
    if len(valid_columns) <= 1:
        print("⚠️  업로드할 컬럼이 없습니다. (date만 있음)")
        cursor.close()
        conn.close()
        return
    
    # SQL 쿼리 동적 생성
    placeholders = ",".join(["%s"] * len(valid_columns))
    columns_str = ",".join([f"`{col}`" for col in valid_columns])
    sql = f"INSERT INTO eiExchangeRate ({columns_str}) VALUES ({placeholders})"
    
    print(f"업로드할 컬럼: {valid_columns}")
    print(f"데이터 업로드 중... (총 {len(df_to_insert)}행)")
    
    for idx, (_, row) in enumerate(df_to_insert.iterrows(), 1):
        # NaN을 None으로 변환
        values = tuple(row.replace({np.nan: None}))
        cursor.execute(sql, values)
        if idx % 100 == 0:
            print(f"  진행: {idx}/{len(df_to_insert)}")
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"✅ 데이터베이스 업로드 완료! (총 {len(df_to_insert)}행)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default=None, help="시계열 사양 CSV 경로 (기본값: data/series_specs.csv)")
    ap.add_argument("--start", required=True, help="전역 시작일 YYYY-MM-DD")
    ap.add_argument("--end",   required=True, help="전역 종료일 YYYY-MM-DD")
    ap.add_argument("--out",   default=None, help="최종 저장 경로(CSV) (기본값: data/ecos.csv)")
    ap.add_argument("--db",    action="store_true", help="CSV 저장 후 MariaDB에 업로드")
    args = ap.parse_args()

    # series_specs.csv 경로 결정 (프로젝트 루트 기준)
    if args.specs:
        specs_path = Path(args.specs)
        if not specs_path.is_absolute():
            specs_path = project_root / specs_path
    else:
        # 기본 경로: 프로젝트 루트의 data/series_specs.csv
        specs_path = project_root / "data" / "series_specs.csv"
    
    if not specs_path.exists():
        raise FileNotFoundError(f"사양 파일을 찾을 수 없습니다: {specs_path}")
    
    specs = load_specs(str(specs_path))
    
    # output 경로 결정 (프로젝트 루트 기준)
    if args.out:
        output_path = Path(args.out)
        if not output_path.is_absolute():
            output_path = project_root / output_path
    else:
        # 기본 경로: 프로젝트 루트의 data/ecos.csv
        output_path = project_root / "data" / "ecos.csv"
    
    # 출력 디렉토리 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)

    series_wide_inputs: List[Tuple[str, pd.DataFrame]] = []

    for i, sp in enumerate(specs, 1):
        stat = sp.stat_code
        cyc  = sp.cycle.upper()
        name = sp.name if sp.name else f"value_{stat}"

        print(f"[{i}/{len(specs)}] {stat} ({cyc}) {args.start}~{args.end} 수집 중...")

        # 1) 원자료 수집
        raw = fetch_auto(stat_code=stat, cycle=cyc, item_code1=sp.item_code1, item_code2=sp.item_code2,
                         start=args.start, end=args.end)

        # 2) ['date','value']로 축소
        slim = to_value_frame(raw, cyc)

        # 3) 월/분기/연 값을 기간 시작일로 정규화(월말값 → 월초값 등)
        slim = normalize_date_by_cycle(slim, cyc)

        # 4) 일단위로 확장(각 기간의 상수로 퍼뜨림; 내부 ffill 사용)
        daily = expand_to_daily(slim, cyc, args.start, args.end)

        series_wide_inputs.append((name, daily))

    # 5) 와이드 병합
    wide = merge_wide(series_wide_inputs)

    # 6) 안전장치: 전역 범위 일단위 재인덱싱 후 ffill로 빈칸 보정
    wide["date"] = pd.to_datetime(wide["date"])
    wide = wide.sort_values("date")
    idx = pd.date_range(start=args.start, end=args.end, freq="D")
    wide = (wide.set_index("date")
                 .reindex(idx)
                 .ffill()
                 .rename_axis("date")
                 .reset_index())

    # 7) 저장
    wide["date"] = wide["date"].dt.strftime("%Y-%m-%d")
    wide.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ CSV 저장 완료: {output_path}")
    
    # 8) DB 업로드 (옵션)
    if args.db:
        print("\n" + "="*50)
        print("MariaDB 업로드 시작")
        print("="*50)
        upload_to_db(wide)

if __name__ == "__main__":
    main()
