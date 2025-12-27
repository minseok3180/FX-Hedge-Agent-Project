"""날짜 파싱 도구 - 다양한 날짜 형식을 표준 형식(YYYY-MM-DD)으로 변환"""
import re
from typing import Optional
from datetime import datetime
from src.utils.logger import get_logger
from src.utils.tools import tool, handle_tool_error, ToolError

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    def Field(*args, **kwargs):
        return None

logger = get_logger("date-parser-tool")


class DateParseInput(BaseModel):
    """날짜 파싱 입력 스키마"""
    date_string: str = Field(
        description="파싱할 날짜 문자열 (다양한 형식 지원)",
        examples=["10/02", "10월 2일", "2025-10-02", "10-02", "10.02"]
    )
    default_year: int = Field(
        default=2025,
        description="연도가 없을 때 사용할 기본 연도",
        examples=[2025]
    )


@tool(args_schema=DateParseInput)
@handle_tool_error("date_parse")
async def date_parse(date_string: str, default_year: int = 2025) -> dict:
    """
    다양한 날짜 형식을 표준 형식(YYYY-MM-DD)으로 변환합니다.
    
    지원하는 형식:
    - "10/02" -> "2025-10-02"
    - "10월 2일" -> "2025-10-02"
    - "2025-10-02" -> "2025-10-02" (그대로 반환)
    - "10-02" -> "2025-10-02"
    - "10.02" -> "2025-10-02"
    - "10/2" -> "2025-10-02"
    - "10월 2일" -> "2025-10-02"
    
    Args:
        date_string: 파싱할 날짜 문자열
        default_year: 연도가 없을 때 사용할 기본 연도 (기본값: 2025)
    
    Returns:
        {
          "status": "success",
          "parsed_date": "YYYY-MM-DD 형식의 날짜",
          "original": "원본 문자열"
        }
    """
    logger.info(
        "🔧 [TOOL CALL] date_parse 실행",
        {"tool_name": "date_parse", "date_string": date_string, "default_year": default_year},
    )
    
    original = date_string.strip()
    
    try:
        # 이미 YYYY-MM-DD 형식인 경우 - 연도가 명시되어 있으면 그대로 사용
        if re.match(r'^\d{4}-\d{2}-\d{2}$', original):
            logger.info(
                "✅ [DATE PARSE] 이미 표준 형식 (연도 유지)",
                {"original": original, "parsed": original}
            )
            return {
                "status": "success",
                "parsed_date": original,
                "original": original
            }
        
        # MM/DD 형식 (예: 10/02, 10/2)
        match = re.match(r'^(\d{1,2})/(\d{1,2})$', original)
        if match:
            month = int(match.group(1))
            day = int(match.group(2))
            parsed = f"{default_year}-{month:02d}-{day:02d}"
            logger.info(
                "✅ [DATE PARSE] MM/DD 형식 파싱",
                {"original": original, "parsed": parsed}
            )
            return {
                "status": "success",
                "parsed_date": parsed,
                "original": original
            }
        
        # MM-DD 형식 (예: 10-02, 10-2)
        match = re.match(r'^(\d{1,2})-(\d{1,2})$', original)
        if match:
            month = int(match.group(1))
            day = int(match.group(2))
            parsed = f"{default_year}-{month:02d}-{day:02d}"
            logger.info(
                "✅ [DATE PARSE] MM-DD 형식 파싱",
                {"original": original, "parsed": parsed}
            )
            return {
                "status": "success",
                "parsed_date": parsed,
                "original": original
            }
        
        # MM.DD 형식 (예: 10.02, 10.2)
        match = re.match(r'^(\d{1,2})\.(\d{1,2})$', original)
        if match:
            month = int(match.group(1))
            day = int(match.group(2))
            parsed = f"{default_year}-{month:02d}-{day:02d}"
            logger.info(
                "✅ [DATE PARSE] MM.DD 형식 파싱",
                {"original": original, "parsed": parsed}
            )
            return {
                "status": "success",
                "parsed_date": parsed,
                "original": original
            }
        
        # MM월 DD일 형식 (예: 10월 2일, 10월 02일)
        match = re.match(r'^(\d{1,2})월\s*(\d{1,2})일$', original)
        if match:
            month = int(match.group(1))
            day = int(match.group(2))
            parsed = f"{default_year}-{month:02d}-{day:02d}"
            logger.info(
                "✅ [DATE PARSE] MM월 DD일 형식 파싱",
                {"original": original, "parsed": parsed}
            )
            return {
                "status": "success",
                "parsed_date": parsed,
                "original": original
            }
        
        # YYYY/MM/DD 형식 (예: 2025/10/02) - 연도가 명시되어 있으면 그대로 사용
        match = re.match(r'^(\d{4})/(\d{1,2})/(\d{1,2})$', original)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            parsed = f"{year}-{month:02d}-{day:02d}"
            logger.info(
                "✅ [DATE PARSE] YYYY/MM/DD 형식 파싱 (연도 유지)",
                {"original": original, "parsed": parsed}
            )
            return {
                "status": "success",
                "parsed_date": parsed,
                "original": original
            }
        
        # 파싱 실패
        logger.warning(
            "⚠️ [DATE PARSE] 지원하지 않는 형식",
            {"original": original}
        )
        return {
            "status": "error",
            "parsed_date": None,
            "original": original,
            "error": f"지원하지 않는 날짜 형식: {original}"
        }
        
    except Exception as e:
        logger.error(
            "❌ date_parse 실패",
            {"date_string": date_string, "error": str(e)},
            exc_info=True,
        )
        raise ToolError("date_parse", f"날짜 파싱 실패: {str(e)}", e)

