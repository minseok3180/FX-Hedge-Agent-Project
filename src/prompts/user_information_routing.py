"""User Information 에이전트 라우팅 설명 (Supervisor 라우팅용)"""

USER_INFORMATION_ROUTING = """## 에이전트: user_information

**역할**: 사용자 프로필 및 자산 정보를 조회·관리하는 에이전트입니다. 
MariaDB의 user_info 테이블에 저장된 정보를 조회하거나 입력/수정(Upsert)합니다.

**보유 툴**:
- user_info_get: user_id를 기준으로 사용자 정보를 조회
- user_info_upsert: 이름, 나이, 성별, 총재산, 해외재산, 투자성향을 포함한 사용자 정보를 입력/수정

**사용 시기**: 
- 특정 사용자(user_id)의 기본 정보와 자산 현황이 필요할 때
- 다른 에이전트가 사용자 투자성향(risk_profile)이나 자산 규모(total_assets, overseas_assets)를 참고해야 할 때
- 사용자가 자신의 정보를 새로 입력하거나 수정해 달라고 요청할 때
"""


