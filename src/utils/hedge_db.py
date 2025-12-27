"""
헷지 관련 DB 유틸리티 함수
"""
import os
import json
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
import pymysql
from dotenv import load_dotenv

load_dotenv()


def get_db_connection():
    """DB 연결 반환"""
    db_host = os.getenv("DATABASE_HOST") or os.getenv("DB_HOST")
    db_port = int(os.getenv("DATABASE_PORT") or os.getenv("DB_PORT", 3306))
    db_user = os.getenv("DATABASE_USER") or os.getenv("DB_USER")
    db_password = os.getenv("DATABASE_PASSWORD") or os.getenv("DB_PASSWORD")
    db_name = os.getenv("DATABASE_NAME") or os.getenv("DB_NAME")

    if not all([db_host, db_user, db_password, db_name]):
        raise ValueError("데이터베이스 연결 정보가 없습니다.")

    return pymysql.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        database=db_name,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def ensure_hedge_tables():
    """
    헷지 관련 테이블이 없으면 생성하고, 필요한 컬럼이 없으면 추가하는 공통 유틸 함수
    
    Returns:
        bool: 성공 여부
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # user_info 테이블의 user_id 타입 확인
        cursor.execute("DESCRIBE user_info")
        user_info_columns = {col["Field"]: col["Type"] for col in cursor.fetchall()}
        user_id_type = user_info_columns.get("user_id", "VARCHAR(10)")
        
        # user_info 테이블의 charset/collation 확인
        cursor.execute("SHOW CREATE TABLE user_info")
        create_user_info = cursor.fetchone()
        user_info_charset = "utf8mb4"
        user_info_collation = "utf8mb4_general_ci"
        if create_user_info:
            create_sql = create_user_info.get("Create Table", "")
            if "COLLATE=" in create_sql:
                import re
                collation_match = re.search(r"COLLATE=([^\s]+)", create_sql)
                if collation_match:
                    user_info_collation = collation_match.group(1)

        # (1) user_hedge_settings 테이블 생성
        create_settings_table_sql = f"""
        CREATE TABLE IF NOT EXISTS user_hedge_settings (
            user_id {user_id_type} PRIMARY KEY COMMENT '사용자 ID',
            hedge_mode ENUM('manual', 'auto', 'recommend_only') DEFAULT 'manual' COMMENT '헷지 모드',
            confirmed_hedge_ratio DECIMAL(6, 5) NULL COMMENT '확정된 헷지 비율',
            rebalance_threshold DECIMAL(6, 5) DEFAULT 0.05000 COMMENT '리밸런싱 임계값',
            total_overseas_usd DECIMAL(15, 2) NULL COMMENT '해외자산 총액 (USD)',
            hedged_etf DECIMAL(15, 2) NULL COMMENT '헷지된 ETF 금액 (USD)',
            unhedged_etf DECIMAL(15, 2) NULL COMMENT '비헷지 ETF 금액 (USD)',
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '수정 시간',
            
            CONSTRAINT fk_hedge_settings_user_id 
                FOREIGN KEY (user_id) 
                REFERENCES user_info(user_id) 
                ON DELETE CASCADE 
                ON UPDATE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET={user_info_charset} COLLATE={user_info_collation}
        COMMENT='사용자 헷지 설정 (현재 설정)';
        """
        
        cursor.execute(create_settings_table_sql)
        
        # user_hedge_settings에 컬럼 추가 (이미 테이블이 있는 경우)
        cursor.execute("DESCRIBE user_hedge_settings")
        settings_columns = {col["Field"]: col["Type"] for col in cursor.fetchall()}
        
        if "total_overseas_usd" not in settings_columns:
            cursor.execute("ALTER TABLE user_hedge_settings ADD COLUMN total_overseas_usd DECIMAL(15, 2) NULL COMMENT '해외자산 총액 (USD)' AFTER rebalance_threshold")
        if "hedged_etf" not in settings_columns:
            cursor.execute("ALTER TABLE user_hedge_settings ADD COLUMN hedged_etf DECIMAL(15, 2) NULL COMMENT '헷지된 ETF 금액 (USD)' AFTER total_overseas_usd")
        if "unhedged_etf" not in settings_columns:
            cursor.execute("ALTER TABLE user_hedge_settings ADD COLUMN unhedged_etf DECIMAL(15, 2) NULL COMMENT '비헷지 ETF 금액 (USD)' AFTER hedged_etf")

        # (2) user_hedge_ratio_history 테이블 생성
        create_history_table_sql = f"""
        CREATE TABLE IF NOT EXISTS user_hedge_ratio_history (
            id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '로그 ID',
            user_id {user_id_type} NOT NULL COMMENT '사용자 ID',
            event_type VARCHAR(32) NOT NULL COMMENT '이벤트 타입',
            ratio DECIMAL(6, 5) NULL COMMENT '헷지 비율',
            source VARCHAR(32) NOT NULL COMMENT '소스 (MODEL, USER, SYSTEM)',
            run_id VARCHAR(64) NOT NULL COMMENT '실행 ID',
            total_overseas_usd DECIMAL(15, 2) NULL COMMENT '해외자산 총액 (USD)',
            hedged_etf DECIMAL(15, 2) NULL COMMENT '헷지된 ETF 금액 (USD)',
            unhedged_etf DECIMAL(15, 2) NULL COMMENT '비헷지 ETF 금액 (USD)',
            user_message TEXT NULL COMMENT '사용자 메시지',
            computed_json JSON NULL COMMENT '계산 결과 JSON',
            decision_json JSON NULL COMMENT '의사결정 JSON',
            status VARCHAR(16) DEFAULT 'success' COMMENT '상태',
            error_message TEXT NULL COMMENT '에러 메시지',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '생성 시간',
            
            CONSTRAINT fk_hedge_history_user_id 
                FOREIGN KEY (user_id) 
                REFERENCES user_info(user_id) 
                ON DELETE CASCADE 
                ON UPDATE CASCADE,
            
            INDEX idx_user_created (user_id, created_at),
            UNIQUE KEY uk_user_run_event (user_id, run_id, event_type),
            
            CHECK (event_type IN ('RECOMMENDED', 'CONFIRM_REQUESTED', 'CONFIRMED', 'OVERRIDDEN', 'REJECTED', 'DECLINED', 'ERROR')),
            CHECK (source IN ('MODEL', 'USER', 'SYSTEM')),
            CHECK (ratio IS NULL OR (ratio >= 0 AND ratio <= 1))
        ) ENGINE=InnoDB DEFAULT CHARSET={user_info_charset} COLLATE={user_info_collation}
        COMMENT='헷지 비율 히스토리 로그';
        """
        
        cursor.execute(create_history_table_sql)
        
        # user_hedge_ratio_history에 컬럼 추가 (이미 테이블이 있는 경우)
        cursor.execute("DESCRIBE user_hedge_ratio_history")
        history_columns = {col["Field"]: col["Type"] for col in cursor.fetchall()}
        
        if "total_overseas_usd" not in history_columns:
            cursor.execute("ALTER TABLE user_hedge_ratio_history ADD COLUMN total_overseas_usd DECIMAL(15, 2) NULL COMMENT '해외자산 총액 (USD)' AFTER run_id")
        if "hedged_etf" not in history_columns:
            cursor.execute("ALTER TABLE user_hedge_ratio_history ADD COLUMN hedged_etf DECIMAL(15, 2) NULL COMMENT '헷지된 ETF 금액 (USD)' AFTER total_overseas_usd")
        if "unhedged_etf" not in history_columns:
            cursor.execute("ALTER TABLE user_hedge_ratio_history ADD COLUMN unhedged_etf DECIMAL(15, 2) NULL COMMENT '비헷지 ETF 금액 (USD)' AFTER hedged_etf")

        conn.commit()
        conn.close()
        
        return True

    except Exception as e:
        print(f"❌ 테이블 생성 오류: {e}")
        return False


def log_hedge_event(
    user_id: str,
    event_type: str,
    run_id: str,
    source: str,
    ratio: Optional[float] = None,
    total_overseas_usd: Optional[float] = None,
    hedged_etf: Optional[float] = None,
    unhedged_etf: Optional[float] = None,
    user_message: Optional[str] = None,
    computed_json: Optional[Dict[str, Any]] = None,
    decision_json: Optional[Dict[str, Any]] = None,
    status: str = "success",
    error_message: Optional[str] = None,
) -> bool:
    """
    헷지 이벤트를 히스토리에 기록
    
    Args:
        user_id: 사용자 ID
        event_type: 이벤트 타입 ('RECOMMENDED', 'CONFIRM_REQUESTED', 'CONFIRMED', 'OVERRIDDEN', 'REJECTED', 'ERROR')
        run_id: 실행 ID (UUID)
        source: 소스 ('MODEL', 'USER', 'SYSTEM')
        ratio: 헷지 비율 (선택)
        total_overseas_usd: 해외자산 총액 USD (선택)
        hedged_etf: 헷지된 ETF 금액 USD (선택)
        unhedged_etf: 비헷지 ETF 금액 USD (선택)
        user_message: 사용자 메시지 (선택)
        computed_json: 계산 결과 JSON (선택)
        decision_json: 의사결정 JSON (선택)
        status: 상태 (기본값: 'success')
        error_message: 에러 메시지 (선택)
        
    Returns:
        bool: 성공 여부
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO user_hedge_ratio_history 
            (user_id, event_type, ratio, source, run_id, total_overseas_usd, hedged_etf, unhedged_etf, user_message, computed_json, decision_json, status, error_message)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                user_id,
                event_type,
                ratio,
                source,
                run_id,
                total_overseas_usd,
                hedged_etf,
                unhedged_etf,
                user_message,
                json.dumps(computed_json) if computed_json else None,
                json.dumps(decision_json) if decision_json else None,
                status,
                error_message,
            ),
        )

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"❌ 이벤트 로깅 오류: {e}")
        return False


def update_hedge_settings(
    user_id: str,
    confirmed_hedge_ratio: Optional[float] = None,
    hedge_mode: Optional[str] = None,
    total_overseas_usd: Optional[float] = None,
    hedged_etf: Optional[float] = None,
    unhedged_etf: Optional[float] = None,
) -> bool:
    """
    사용자 헷지 설정 업데이트 (upsert)
    
    Args:
        user_id: 사용자 ID
        confirmed_hedge_ratio: 확정된 헷지 비율 (선택)
        hedge_mode: 헷지 모드 (선택)
        total_overseas_usd: 해외자산 총액 USD (선택)
        hedged_etf: 헷지된 ETF 금액 USD (선택)
        unhedged_etf: 비헷지 ETF 금액 USD (선택)
        
    Returns:
        bool: 성공 여부
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # UPSERT: ON DUPLICATE KEY UPDATE 사용
        update_fields = []
        values = [user_id]

        if confirmed_hedge_ratio is not None:
            update_fields.append("confirmed_hedge_ratio")
            values.append(confirmed_hedge_ratio)

        if hedge_mode is not None:
            update_fields.append("hedge_mode")
            values.append(hedge_mode)
            
        if total_overseas_usd is not None:
            update_fields.append("total_overseas_usd")
            values.append(total_overseas_usd)
            
        if hedged_etf is not None:
            update_fields.append("hedged_etf")
            values.append(hedged_etf)
            
        if unhedged_etf is not None:
            update_fields.append("unhedged_etf")
            values.append(unhedged_etf)

        if not update_fields:
            # 업데이트할 필드가 없으면 기본값으로 INSERT만
            cursor.execute(
                """
                INSERT INTO user_hedge_settings (user_id)
                VALUES (%s)
                ON DUPLICATE KEY UPDATE updated_at = CURRENT_TIMESTAMP
                """,
                (user_id,),
            )
        else:
            # INSERT ... ON DUPLICATE KEY UPDATE 구문
            insert_fields = ", ".join(update_fields)
            insert_placeholders = ", ".join(["%s"] * len(update_fields))
            update_clause = ", ".join([f"{field} = VALUES({field})" for field in update_fields])
            
            sql = f"""
            INSERT INTO user_hedge_settings (user_id, {insert_fields})
            VALUES (%s, {insert_placeholders})
            ON DUPLICATE KEY UPDATE {update_clause}, updated_at = CURRENT_TIMESTAMP
            """
            cursor.execute(sql, values)

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"❌ 설정 업데이트 오류: {e}")
        return False


def update_hedge_settings_with_transaction(
    user_id: str,
    confirmed_hedge_ratio: Optional[float] = None,
    hedge_mode: Optional[str] = None,
) -> bool:
    """
    트랜잭션으로 사용자 헷지 설정 업데이트 (upsert)
    트랜잭션을 명시적으로 관리하는 버전
    
    Args:
        user_id: 사용자 ID
        confirmed_hedge_ratio: 확정된 헷지 비율 (선택)
        hedge_mode: 헷지 모드 (선택)
        
    Returns:
        bool: 성공 여부
    """
    return update_hedge_settings(user_id, confirmed_hedge_ratio, hedge_mode)


def get_hedge_settings(user_id: str) -> Optional[Dict[str, Any]]:
    """
    사용자 헷지 설정 조회
    
    Args:
        user_id: 사용자 ID
        
    Returns:
        설정 딕셔너리 또는 None
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT user_id, hedge_mode, confirmed_hedge_ratio, rebalance_threshold, updated_at
            FROM user_hedge_settings
            WHERE user_id = %s
            """,
            (user_id,),
        )

        result = cursor.fetchone()
        conn.close()
        return dict(result) if result else None

    except Exception as e:
        print(f"❌ 설정 조회 오류: {e}")
        return None
