"""
로컬 ChromaDB 관리 모듈
VDB 대신 로컬 ChromaDB를 사용하여 벡터 데이터 관리
"""

import os
import logging
from typing import Dict, List, Any, Optional
import chromadb
from chromadb.config import Settings
import json
from datetime import datetime

from config import Config


class LocalChromaDBManager:
    """로컬 ChromaDB 관리자"""
    
    def __init__(self, config: Config):
        self.config = config.db
        self.logger = logging.getLogger(__name__)
        
        # ChromaDB 클라이언트 초기화
        self.client = None
        self.collection = None
        
        # TODO: ChromaDB 초기화
        self._initialize_chromadb()
    
    def _initialize_chromadb(self):
        """ChromaDB 초기화"""
        try:
            # 데이터 디렉토리 생성
            os.makedirs(self.config.vdb_path, exist_ok=True)
            
            # ChromaDB 클라이언트 생성
            self.client = chromadb.PersistentClient(
                path=self.config.vdb_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 컬렉션 생성 또는 가져오기
            try:
                self.collection = self.client.get_collection(
                    name=self.config.vdb_collection
                )
                self.logger.info(f"기존 컬렉션 로드: {self.config.vdb_collection}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.config.vdb_collection,
                    metadata={"description": "외환 헷지전략 정보 저장소"}
                )
                self.logger.info(f"새 컬렉션 생성: {self.config.vdb_collection}")
            
            # TODO: 샘플 데이터 추가 (개발용)
            self._add_sample_data()
            
        except Exception as e:
            self.logger.error(f"ChromaDB 초기화 실패: {str(e)}")
            raise e
    
    def _add_sample_data(self):
        """샘플 헷지 전략 데이터 추가"""
        try:
            # 기존 데이터 확인
            existing_count = self.collection.count()
            if existing_count > 0:
                self.logger.info(f"기존 데이터 {existing_count}개 발견")
                return
            
            # 샘플 헷지 전략 데이터
            sample_strategies = [
                {
                    "id": "strategy_001",
                    "name": "Forward Hedge",
                    "description": "선물 계약을 통한 외환 헷지 전략",
                    "content": "Forward Hedge는 미래의 특정 날짜에 미리 정해진 환율로 외환을 거래하는 계약을 통해 환율 변동 위험을 헷지하는 전략입니다. 이 전략은 환율 변동에 대한 예측 가능성을 제공하며, 비즈니스 계획 수립에 도움이 됩니다.",
                    "currency_pairs": ["USD/KRW", "EUR/KRW"],
                    "risk_level": "low",
                    "complexity": "simple",
                    "min_amount": 10000,
                    "max_amount": 1000000,
                    "hedge_ratio": 1.0,
                    "expected_return": 0.02,
                    "max_drawdown": 0.05
                },
                {
                    "id": "strategy_002", 
                    "name": "Option Hedge",
                    "description": "옵션을 활용한 외환 헷지 전략",
                    "content": "Option Hedge는 외환 옵션을 사용하여 환율 변동 위험을 제한하면서도 상승 가능성을 유지하는 전략입니다. Put 옵션을 매수하여 하락 위험을 제한하고, Call 옵션을 매도하여 프리미엄을 받을 수 있습니다.",
                    "currency_pairs": ["USD/KRW", "EUR/KRW", "JPY/KRW"],
                    "risk_level": "medium",
                    "complexity": "intermediate",
                    "min_amount": 50000,
                    "max_amount": 500000,
                    "hedge_ratio": 0.8,
                    "expected_return": 0.05,
                    "max_drawdown": 0.10
                },
                {
                    "id": "strategy_003",
                    "name": "Currency Swap",
                    "description": "통화 스왑을 통한 헷지 전략",
                    "content": "Currency Swap은 서로 다른 통화의 이자율과 원금을 교환하는 계약을 통해 환율 위험을 헷지하는 전략입니다. 이 전략은 장기간의 환율 변동 위험을 효과적으로 관리할 수 있습니다.",
                    "currency_pairs": ["USD/KRW", "EUR/KRW", "GBP/KRW"],
                    "risk_level": "low",
                    "complexity": "advanced",
                    "min_amount": 100000,
                    "max_amount": 10000000,
                    "hedge_ratio": 1.0,
                    "expected_return": 0.03,
                    "max_drawdown": 0.08
                },
                {
                    "id": "strategy_004",
                    "name": "Natural Hedge",
                    "description": "자연 헷지를 통한 환율 위험 관리",
                    "content": "Natural Hedge는 비즈니스 운영에서 자연스럽게 발생하는 외화 자산과 부채를 균형있게 관리하여 환율 위험을 최소화하는 전략입니다. 이는 추가적인 금융 상품 없이도 환율 위험을 관리할 수 있는 방법입니다.",
                    "currency_pairs": ["USD/KRW", "EUR/KRW", "JPY/KRW", "CNY/KRW"],
                    "risk_level": "low",
                    "complexity": "simple",
                    "min_amount": 0,
                    "max_amount": 0,
                    "hedge_ratio": 0.5,
                    "expected_return": 0.01,
                    "max_drawdown": 0.03
                }
            ]
            
            # 샘플 데이터를 ChromaDB에 추가
            documents = []
            metadatas = []
            ids = []
            
            for strategy in sample_strategies:
                documents.append(strategy["content"])
                metadatas.append({
                    "name": strategy["name"],
                    "description": strategy["description"],
                    "currency_pairs": json.dumps(strategy["currency_pairs"]),
                    "risk_level": strategy["risk_level"],
                    "complexity": strategy["complexity"],
                    "min_amount": strategy["min_amount"],
                    "max_amount": strategy["max_amount"],
                    "hedge_ratio": strategy["hedge_ratio"],
                    "expected_return": strategy["expected_return"],
                    "max_drawdown": strategy["max_drawdown"],
                    "created_at": datetime.now().isoformat()
                })
                ids.append(strategy["id"])
            
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"샘플 데이터 {len(sample_strategies)}개 추가 완료")
            
        except Exception as e:
            self.logger.error(f"샘플 데이터 추가 실패: {str(e)}")
    
    def add_strategy(self, strategy_data: Dict[str, Any]) -> bool:
        """
        헷지 전략 추가
        
        Args:
            strategy_data: 전략 데이터
            
        Returns:
            추가 성공 여부
        """
        try:
            strategy_id = strategy_data.get("id", f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # 메타데이터 준비
            metadata = {
                "name": strategy_data.get("name", ""),
                "description": strategy_data.get("description", ""),
                "currency_pairs": json.dumps(strategy_data.get("currency_pairs", [])),
                "risk_level": strategy_data.get("risk_level", "medium"),
                "complexity": strategy_data.get("complexity", "intermediate"),
                "min_amount": strategy_data.get("min_amount", 0),
                "max_amount": strategy_data.get("max_amount", 0),
                "hedge_ratio": strategy_data.get("hedge_ratio", 1.0),
                "expected_return": strategy_data.get("expected_return", 0.0),
                "max_drawdown": strategy_data.get("max_drawdown", 0.0),
                "created_at": datetime.now().isoformat()
            }
            
            # 문서 내용
            content = strategy_data.get("content", strategy_data.get("description", ""))
            
            self.collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[strategy_id]
            )
            
            self.logger.info(f"헷지 전략 추가: {strategy_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"헷지 전략 추가 실패: {str(e)}")
            return False
    
    def search_strategies(self, 
                         query: str, 
                         n_results: int = 5,
                         filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        헷지 전략 검색
        
        Args:
            query: 검색 쿼리
            n_results: 결과 개수
            filters: 필터 조건
            
        Returns:
            검색 결과 리스트
        """
        try:
            # ChromaDB에서 검색
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filters
            )
            
            # 결과 포맷팅
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else 0.0
                }
                
                # JSON 필드 파싱
                if 'currency_pairs' in result['metadata']:
                    try:
                        result['metadata']['currency_pairs'] = json.loads(result['metadata']['currency_pairs'])
                    except json.JSONDecodeError:
                        result['metadata']['currency_pairs'] = []
                
                formatted_results.append(result)
            
            self.logger.info(f"전략 검색 완료: {len(formatted_results)}개 결과")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"전략 검색 실패: {str(e)}")
            return []
    
    def get_strategy_by_id(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        ID로 전략 조회
        
        Args:
            strategy_id: 전략 ID
            
        Returns:
            전략 정보 또는 None
        """
        try:
            results = self.collection.get(ids=[strategy_id])
            
            if not results['ids']:
                return None
            
            result = {
                "id": results['ids'][0],
                "content": results['documents'][0],
                "metadata": results['metadatas'][0]
            }
            
            # JSON 필드 파싱
            if 'currency_pairs' in result['metadata']:
                try:
                    result['metadata']['currency_pairs'] = json.loads(result['metadata']['currency_pairs'])
                except json.JSONDecodeError:
                    result['metadata']['currency_pairs'] = []
            
            return result
            
        except Exception as e:
            self.logger.error(f"전략 조회 실패: {str(e)}")
            return None
    
    def get_all_strategies(self) -> List[Dict[str, Any]]:
        """모든 전략 조회"""
        try:
            results = self.collection.get()
            
            formatted_results = []
            for i in range(len(results['ids'])):
                result = {
                    "id": results['ids'][i],
                    "content": results['documents'][i],
                    "metadata": results['metadatas'][i]
                }
                
                # JSON 필드 파싱
                if 'currency_pairs' in result['metadata']:
                    try:
                        result['metadata']['currency_pairs'] = json.loads(result['metadata']['currency_pairs'])
                    except json.JSONDecodeError:
                        result['metadata']['currency_pairs'] = []
                
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"전략 목록 조회 실패: {str(e)}")
            return []
    
    def update_strategy(self, strategy_id: str, update_data: Dict[str, Any]) -> bool:
        """
        전략 정보 업데이트
        
        Args:
            strategy_id: 전략 ID
            update_data: 업데이트할 데이터
            
        Returns:
            업데이트 성공 여부
        """
        try:
            # 기존 전략 조회
            existing = self.get_strategy_by_id(strategy_id)
            if not existing:
                self.logger.warning(f"전략을 찾을 수 없습니다: {strategy_id}")
                return False
            
            # 메타데이터 업데이트
            metadata = existing['metadata'].copy()
            for key, value in update_data.items():
                if key == 'currency_pairs':
                    metadata[key] = json.dumps(value)
                elif key == 'content':
                    continue  # 문서 내용은 별도 처리
                else:
                    metadata[key] = value
            
            metadata['updated_at'] = datetime.now().isoformat()
            
            # ChromaDB 업데이트 (삭제 후 재추가)
            self.collection.delete(ids=[strategy_id])
            
            content = update_data.get('content', existing['content'])
            
            self.collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[strategy_id]
            )
            
            self.logger.info(f"전략 업데이트 완료: {strategy_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"전략 업데이트 실패: {str(e)}")
            return False
    
    def delete_strategy(self, strategy_id: str) -> bool:
        """
        전략 삭제
        
        Args:
            strategy_id: 전략 ID
            
        Returns:
            삭제 성공 여부
        """
        try:
            self.collection.delete(ids=[strategy_id])
            self.logger.info(f"전략 삭제 완료: {strategy_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"전략 삭제 실패: {str(e)}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """컬렉션 정보 반환"""
        try:
            count = self.collection.count()
            
            return {
                "collection_name": self.config.vdb_collection,
                "total_strategies": count,
                "db_path": self.config.vdb_path,
                "status": "active"
            }
            
        except Exception as e:
            self.logger.error(f"컬렉션 정보 조회 실패: {str(e)}")
            return {
                "collection_name": self.config.vdb_collection,
                "total_strategies": 0,
                "db_path": self.config.vdb_path,
                "status": "error"
            }
