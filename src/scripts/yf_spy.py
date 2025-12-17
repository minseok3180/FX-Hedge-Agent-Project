#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Yahoo Finance API를 이용하여 SPY 종가를 MariaDB에 추가하는 스크립트.

기존 eiExchangeRate 테이블에 SPY_close 컬럼을 추가하고,
해당 테이블의 date 컬럼에 해당하는 날짜에 맞춰 데이터를 채웁니다.

실행 방법:
    py -m src.scripts.yf_spy
    또는
    python3 -m src.scripts.yf_spy
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import pymysql
import logging
from dotenv import load_dotenv

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

# 간단한 로거 설정 (settings에 의존하지 않음)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("yf-spy-tool")

# MariaDB 설정 (.env에서 직접 가져오기)
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
    raise ValueError(
        "MariaDB 연결 정보가 설정되지 않았습니다. "
        ".env 파일에 DB_HOST, DB_NAME, DB_USER, DB_PASSWORD를 추가하세요."
    )

TABLE_NAME = "eiExchangeRate"
DATE_COLUMN = "date"
COLUMN_NAME = "SPY_close"
TICKER_SYMBOL = "SPY"


# ---------------------------------------------------------------------------
# Yahoo Finance API 호출
# ---------------------------------------------------------------------------

def fetch_spy_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Yahoo Finance API를 사용하여 SPY 종가 데이터를 가져옵니다.
    
    Args:
        start_date: 시작 날짜 (YYYY-MM-DD)
        end_date: 종료 날짜 (YYYY-MM-DD)
        
    Returns:
        ['date', 'value'] 형태의 DataFrame
    """
    try:
        import yfinance as yf
        
        logger.info(f"[Yahoo Finance] {TICKER_SYMBOL} 종가 데이터 수집 중... ({start_date} ~ {end_date})")
        
        # Yahoo Finance에서 데이터 가져오기
        ticker = yf.Ticker(TICKER_SYMBOL)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            logger.warning(f"[Yahoo Finance] {TICKER_SYMBOL}에 대한 데이터가 없습니다.")
            return pd.DataFrame(columns=["date", "value"])
        
        # Close 가격만 추출하고 날짜를 인덱스에서 컬럼으로 변환
        df = df[["Close"]].copy()
        df.reset_index(inplace=True)
        df.rename(columns={"Date": "date", "Close": "value"}, inplace=True)
        
        # 날짜를 datetime으로 변환
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["date"] = pd.to_datetime(df["date"])
        
        # 결측치 제거
        df = df.dropna(subset=["value"])
        
        if df.empty:
            logger.warning(f"[Yahoo Finance] {TICKER_SYMBOL}에 대한 유효한 데이터가 없습니다.")
            return pd.DataFrame(columns=["date", "value"])
        
        df = df.sort_values("date").reset_index(drop=True)
        logger.info(f"[Yahoo Finance] {TICKER_SYMBOL} 데이터 수집 완료: {len(df)}개 행")
        
        return df
        
    except ImportError:
        logger.error("yfinance 라이브러리가 필요합니다. pip install yfinance")
        raise
    except Exception as e:
        logger.error(f"[Yahoo Finance] {TICKER_SYMBOL} 데이터 수집 실패: {str(e)}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# MariaDB 작업
# ---------------------------------------------------------------------------

def get_db_connection():
    """
    MariaDB 연결을 생성하고 반환합니다.
    
    Returns:
        pymysql.Connection 객체
    """
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False
    )


def get_existing_dates() -> pd.DataFrame:
    """
    MariaDB의 eiExchangeRate 테이블에서 기존 날짜 목록을 가져옵니다.
    
    Returns:
        ['date'] 형태의 DataFrame
    """
    logger.info("데이터베이스에서 기존 날짜 목록 조회 중...")
    conn = get_db_connection()
    
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT DISTINCT `{DATE_COLUMN}` FROM `{TABLE_NAME}` ORDER BY `{DATE_COLUMN}`")
            rows = cursor.fetchall()
            
            if not rows:
                logger.warning(f"{TABLE_NAME} 테이블에 데이터가 없습니다.")
                return pd.DataFrame(columns=["date"])
            
            dates = [row[DATE_COLUMN] if isinstance(row, dict) else row[0] for row in rows]
            df = pd.DataFrame({"date": pd.to_datetime(dates)})
            
            logger.info(f"기존 날짜 {len(df)}개 발견: {df['date'].min()} ~ {df['date'].max()}")
            return df
            
    except Exception as e:
        logger.error(f"날짜 목록 조회 실패: {str(e)}", exc_info=True)
        raise
    finally:
        conn.close()


def ensure_column_exists() -> None:
    """
    테이블에 SPY_close 컬럼이 존재하는지 확인하고, 없으면 추가합니다.
    """
    logger.info(f"컬럼 존재 여부 확인 및 추가: {COLUMN_NAME}")
    conn = get_db_connection()
    
    try:
        with conn.cursor() as cursor:
            # 기존 컬럼 목록 조회
            cursor.execute(f"DESCRIBE `{TABLE_NAME}`")
            rows = cursor.fetchall()
            existing_columns = {
                row["Field"] if isinstance(row, dict) else row[0]
                for row in rows
            }
            
            # 컬럼이 없으면 추가
            if COLUMN_NAME not in existing_columns:
                logger.info(f"컬럼 추가 중: {COLUMN_NAME}")
                alter_sql = f"ALTER TABLE `{TABLE_NAME}` ADD COLUMN `{COLUMN_NAME}` DOUBLE NULL"
                cursor.execute(alter_sql)
                conn.commit()
                logger.info(f"컬럼 추가 완료: {COLUMN_NAME}")
            else:
                logger.info(f"컬럼 {COLUMN_NAME}이 이미 존재합니다.")
            
    except Exception as e:
        logger.error(f"컬럼 추가 실패: {str(e)}", exc_info=True)
        conn.rollback()
        raise
    finally:
        conn.close()


def update_spy_data(df_dates: pd.DataFrame, df_spy: pd.DataFrame) -> Dict[str, Any]:
    """
    SPY 데이터를 MariaDB에 업데이트합니다.
    
    Args:
        df_dates: 기존 날짜 DataFrame
        df_spy: SPY 데이터 DataFrame
        
    Returns:
        업데이트 결과 딕셔너리
    """
    if df_spy.empty:
        return {"success": False, "message": "업데이트할 데이터가 없습니다."}
    
    logger.info("=" * 60)
    logger.info("SPY 데이터 업데이트 시작")
    logger.info("=" * 60)
    
    conn = get_db_connection()
    update_count = 0
    
    try:
        with conn.cursor() as cursor:
            # 날짜 기준으로 병합
            df_merged = df_dates.merge(
                df_spy,
                on="date",
                how="left",
                suffixes=("", "_spy")
            )
            
            # 업데이트할 행 수 계산
            valid_rows = df_merged["value"].notna()
            update_count = valid_rows.sum()
            
            if update_count == 0:
                logger.warning(f"{COLUMN_NAME}: 매칭되는 데이터가 없습니다.")
                return {"success": False, "update_count": 0}
            
            logger.info(f"{COLUMN_NAME}: {update_count}개 행 업데이트 중...")
            
            # 배치 업데이트
            updated = 0
            for _, row in df_merged[valid_rows].iterrows():
                date_str = row["date"].strftime("%Y-%m-%d")
                value = row["value"]
                
                update_sql = f"""
                    UPDATE `{TABLE_NAME}`
                    SET `{COLUMN_NAME}` = %s
                    WHERE `{DATE_COLUMN}` = %s
                """
                cursor.execute(update_sql, (value, date_str))
                updated += 1
                
                if updated % 100 == 0:
                    logger.debug(f"  진행: {updated}/{update_count}")
            
            conn.commit()
            logger.info(f"{COLUMN_NAME}: {updated}개 행 업데이트 완료")
            logger.info("=" * 60)
            logger.info("SPY 데이터 업데이트 완료")
            logger.info("=" * 60)
            
    except Exception as e:
        logger.error(f"데이터 업데이트 실패: {str(e)}", exc_info=True)
        conn.rollback()
        raise
    finally:
        conn.close()
    
    return {
        "success": True,
        "update_count": updated,
        "total_updated": updated
    }


def update_db_metadata() -> None:
    """
    src/tools/rdb.py의 DB_METADATA에 새로운 컬럼 정보를 추가합니다.
    """
    rdb_file = PROJECT_ROOT / "src" / "tools" / "rdb.py"
    
    if not rdb_file.exists():
        logger.warning(f"rdb.py 파일을 찾을 수 없습니다: {rdb_file}")
        return
    
    logger.info(f"DB_METADATA 업데이트 중: {COLUMN_NAME}")
    
    try:
        with open(rdb_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # 이미 존재하는지 확인
        col_str = f'"{COLUMN_NAME}":'
        if any(col_str in line for line in lines):
            logger.info(f"{COLUMN_NAME}는 이미 DB_METADATA에 존재합니다.")
            return
        
        # "미국금리" 라인 찾기
        insert_index = None
        for i, line in enumerate(lines):
            if '"미국금리":' in line:
                insert_index = i + 1
                break
        
        if insert_index is None:
            logger.warning("DB_METADATA 업데이트 위치를 찾을 수 없습니다. 수동으로 추가해주세요.")
            logger.info(f"추가할 내용:")
            logger.info(f'                "{COLUMN_NAME}": {{"type": "DOUBLE", "description": "SPY 종가"}},')
            return
        
        # 컬럼 엔트리 생성
        col_entry = f'                "{COLUMN_NAME}": {{"type": "DOUBLE", "description": "SPY 종가"}},\n'
        
        # 라인 삽입
        lines[insert_index:insert_index] = [col_entry]
        
        # 파일 저장
        with open(rdb_file, "w", encoding="utf-8") as f:
            f.writelines(lines)
        
        logger.info(f"DB_METADATA에 {COLUMN_NAME} 추가 완료")
        
    except Exception as e:
        logger.warning(f"DB_METADATA 업데이트 실패: {str(e)}")
        logger.info("수동으로 src/tools/rdb.py의 DB_METADATA에 다음을 추가해주세요:")
        logger.info(f'                "{COLUMN_NAME}": {{"type": "DOUBLE", "description": "SPY 종가"}},')


# ---------------------------------------------------------------------------
# 메인 파이프라인
# ---------------------------------------------------------------------------

def run_spy_pipeline() -> Dict[str, Any]:
    """
    SPY 데이터 수집 및 MariaDB 업데이트 파이프라인을 실행합니다.
    
    Returns:
        파이프라인 실행 결과 딕셔너리
    """
    result = {
        "success": False,
        "column_added": False,
        "update_count": 0,
        "errors": []
    }
    
    try:
        # 1) 기존 날짜 목록 가져오기
        df_dates = get_existing_dates()
        
        if df_dates.empty:
            logger.error("업데이트할 날짜가 없습니다.")
            result["errors"].append("업데이트할 날짜가 없습니다.")
            return result
        
        # 날짜 범위 계산
        start_date = df_dates["date"].min().strftime("%Y-%m-%d")
        end_date = df_dates["date"].max().strftime("%Y-%m-%d")
        
        logger.info(f"날짜 범위: {start_date} ~ {end_date}")
        
        # 2) Yahoo Finance에서 SPY 데이터 수집
        try:
            df_spy = fetch_spy_data(start_date, end_date)
            if df_spy.empty:
                logger.error("수집된 SPY 데이터가 없습니다.")
                result["errors"].append("수집된 SPY 데이터가 없습니다.")
                return result
        except Exception as e:
            logger.error(f"SPY 데이터 수집 실패: {str(e)}")
            result["errors"].append(f"SPY 데이터 수집 실패: {str(e)}")
            return result
        
        # 3) 컬럼 추가 (없는 경우)
        ensure_column_exists()
        result["column_added"] = True
        
        # 4) 데이터 업데이트
        update_result = update_spy_data(df_dates, df_spy)
        result["update_count"] = update_result.get("update_count", 0)
        result["success"] = update_result.get("success", False)
        
        # 5) DB_METADATA 업데이트
        update_db_metadata()
        
        logger.info("=" * 60)
        logger.info("파이프라인 실행 완료")
        logger.info("=" * 60)
        logger.info(f"추가된 컬럼: {COLUMN_NAME}")
        logger.info(f"업데이트된 행 수: {result['update_count']}")
        
    except Exception as e:
        logger.error(f"파이프라인 실행 실패: {str(e)}", exc_info=True)
        result["errors"].append(str(e))
        result["success"] = False
    
    return result


def main():
    """CLI 진입점"""
    global TABLE_NAME
    
    parser = argparse.ArgumentParser(
        description="Yahoo Finance API를 사용하여 SPY 종가 데이터를 MariaDB에 추가"
    )
    parser.add_argument(
        "--table",
        default=TABLE_NAME,
        help=f"업데이트할 테이블명 (기본값: {TABLE_NAME})"
    )
    
    args = parser.parse_args()
    TABLE_NAME = args.table
    
    result = run_spy_pipeline()
    
    print("\n" + "=" * 60)
    print("실행 결과:")
    print("=" * 60)
    print(f"성공: {result['success']}")
    print(f"컬럼 추가: {result['column_added']}")
    print(f"업데이트된 행 수: {result['update_count']}")
    if result["errors"]:
        print(f"오류: {result['errors']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

