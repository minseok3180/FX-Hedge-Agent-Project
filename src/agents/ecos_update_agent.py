# src/agents/ecos_update_agent.py
#ECOS API 호출 → 머지 → 일단위 확장 → CSV 저장 → MariaDB 업로드)을 한 번에 실행하는 Collector/ETL용 에이전트.



from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.agents.base_agent import BaseAgent

# 네가 이전에 만들 ecos 파이프라인 모듈을 가정
# (앞에서 내가 제안했던 것처럼 작성했다면 이런 시그니처일 것)
from src.tools.ecos import run_ecos_pipeline


class EcosUpdateAgent(BaseAgent):
    """
    ECOS API에서 지표를 수집하고, 머지/일단위 확장/CSV 저장 후
    MariaDB(eiExchangeRate)에 업로드하는 Collector/ETL 에이전트.

    - 실제 로직은 src/tools/ecos.py의 run_ecos_pipeline을 호출해 수행한다.
    - 테이블 구조는 ecos_main.py와 동일하게, 기존 데이터를 전부 대체한다.
    """

    def __init__(
        self,
        default_specs: Optional[str] = None,
        default_out: Optional[str] = None,
        table_name: str = "eiExchangeRate",
    ):
        """
        Args:
            default_specs: series_specs.csv 경로 (프로젝트 루트 기준 상대경로 가능)
            default_out: CSV 기본 저장 경로
            table_name: 업로드 대상 테이블명
        """
        super().__init__(
            name="ecos_update",
            description="ECOS API에서 지표를 수집해 CSV + MariaDB(eiExchangeRate)에 적재하는 Collector 에이전트",
        )
        self.default_specs = default_specs
        self.default_out = default_out
        self.table_name = table_name

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            task: Supervisor가 넘긴 자연어 명령 (예: "2020~2025년까지 ECOS 데이터 갱신")
                  실제 파라미터는 context에서 받는다.
            context:
                - start_date (str, 필수): "YYYY-MM-DD"
                - end_date   (str, 필수): "YYYY-MM-DD"
                - specs_path (str, 옵션): 시계열 사양 CSV 경로
                - out_path   (str, 옵션): 출력 CSV 경로
                - upload_db  (bool, 옵션): DB 업로드 여부 (기본 True)

        Returns:
            {
              "agent": "ecos_update",
              "task": <원본 task>,
              "status": "success" | "error",
              "message": <텍스트>,
              "meta": {...}   # ecos 파이프라인에서 되돌려준 메타정보(행 개수, 기간 등)
            }
        """
        context = context or {}

        start_date = context.get("start_date")
        end_date = context.get("end_date")

        if not start_date or not end_date:
            return {
                "agent": self.name,
                "task": task,
                "status": "error",
                "message": "start_date와 end_date가 context에 필요합니다. "
                           "예: {'start_date': '2010-01-01', 'end_date': '2025-09-02'}",
            }

        specs_path = context.get("specs_path", self.default_specs)
        out_path = context.get("out_path", self.default_out)
        upload_db = bool(context.get("upload_db", True))

        try:
            # src/tools/ecos.py에서 구현한 메인 파이프라인 호출
            # run_ecos_pipeline은 다음과 유사한 시그니처를 가정:
            #   run_ecos_pipeline(
            #       start: str,
            #       end: str,
            #       specs_path: Optional[str],
            #       out_path: Optional[str],
            #       upload_to_db: bool = False,
            #       table_name: str = "eiExchangeRate",
            #   ) -> Dict[str, Any]
            result: Dict[str, Any] = run_ecos_pipeline(
                start=start_date,
                end=end_date,
                specs_path=specs_path,
                out_path=out_path,
                upload_to_db=upload_db,
                table_name=self.table_name,
            )

            message_lines = [
                f"ECOS 파이프라인 실행 완료.",
                f"- 기간: {start_date} ~ {end_date}",
                f"- CSV 저장: {result.get('csv_path')}",
            ]
            if upload_db:
                message_lines.append(
                    f"- MariaDB 테이블 '{self.table_name}'에 업로드 완료 "
                    f"(총 {result.get('row_count')}행)"
                )

            return {
                "agent": self.name,
                "task": task,
                "status": "success",
                "message": "\n".join(message_lines),
                "meta": result,
            }

        except Exception as e:
            return {
                "agent": self.name,
                "task": task,
                "status": "error",
                "message": f"ECOS 파이프라인 실행 중 오류 발생: {e}",
            }
