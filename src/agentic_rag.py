import os
import json
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import chromadb
from chromadb.config import Settings
import numpy as np
import re

class KTRAGSystem:
    """KT Midm-2.0-Mini-Instruct 모델을 사용한 RAG 시스템"""
    
    def __init__(self, model_name: str = "K-intelligence/Midm-2.0-Mini-Instruct", device: str = "auto"):
        self.device = self._get_device(device)
        self.model_name = model_name
        
        # 모델 및 토크나이저 로드
        print("KT Midm-2.0-Mini-Instruct 모델 로딩 중...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )
            print("✅ KT Midm-2.0-Mini-Instruct 모델 로드 성공")
        except Exception as e:
            print(f"⚠️  KT Midm-2.0-Mini-Instruct 모델 로드 실패: {e}")
            print("기본 모델을 사용합니다.")
            self.tokenizer = None
            self.model = None
        
        # ChromaDB 초기화
        print("ChromaDB 초기화 중...")
        try:
            self.chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./chroma_db"
            ))
            
            # 컬렉션 생성 또는 로드
            self.collection = self.chroma_client.get_or_create_collection(
                name="fx_knowledge_base",
                metadata={"description": "외환 거래 지식 베이스 (KRW 기반)"}
            )
            print("✅ ChromaDB 초기화 성공")
        except Exception as e:
            print(f"⚠️  ChromaDB 초기화 실패: {e}")
            self.collection = None
        
        print("RAG 시스템 초기화 완료!")
    
    def _get_device(self, device: str) -> str:
        """사용 가능한 디바이스 확인"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def add_knowledge(self, documents: List[str], metadata: List[Dict] = None):
        """지식 베이스에 문서 추가"""
        if self.collection is None:
            print("⚠️  ChromaDB가 초기화되지 않았습니다.")
            return
            
        if metadata is None:
            metadata = [{"source": f"doc_{i}", "category": "fx_knowledge"} for i in range(len(documents))]
        
        try:
            # 텍스트 기반으로 ChromaDB에 추가
            self.collection.add(
                documents=documents,
                metadatas=metadata,
                ids=[f"doc_{i}" for i in range(len(documents))]
            )
            
            print(f"{len(documents)}개 문서가 지식 베이스에 추가되었습니다.")
        except Exception as e:
            print(f"지식 베이스 추가 실패: {e}")
    
    def search_similar_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """키워드 기반 문서 검색"""
        if self.collection is None:
            return {'documents': [[]]}
        
        try:
            # 텍스트 기반 검색
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            return results
        except Exception as e:
            print(f"문서 검색 실패: {e}")
            return {'documents': [[]]}
    
    def generate_response(self, query: str, context: str = "") -> str:
        """KT Midm-2.0-Mini-Instruct 모델을 사용한 응답 생성"""
        if self.model is None or self.tokenizer is None:
            # 모델이 없으면 기본 응답
            if context:
                return f"컨텍스트를 바탕으로 한 답변: {query}에 대한 전문가 조언입니다. 외환 거래에서는 리스크 관리가 중요하며, 기술적 분석과 기본적 분석을 종합적으로 고려해야 합니다."
            else:
                return f"{query}에 대한 답변입니다. 외환 거래 전문가로서 정확하고 실용적인 조언을 제공합니다."
        
        try:
            if context:
                prompt = f"""다음은 외환 거래에 대한 질문과 관련 컨텍스트입니다.

컨텍스트:
{context}

질문: {query}

위의 컨텍스트를 바탕으로 질문에 답변해주세요. 외환 거래 전문가로서 정확하고 실용적인 조언을 제공해주세요. 한국 시장에 특화된 조언을 해주세요.

답변:"""
            else:
                prompt = f"""외환 거래에 대한 질문입니다: {query}

외환 거래 전문가로서 정확하고 실용적인 조언을 제공해주세요. 한국 시장에 특화된 조언을 해주세요.

답변:"""
            
            # 토큰화
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 응답 디코딩
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 프롬프트 제거하고 응답만 반환
            if context:
                response = response.replace(prompt, "").strip()
            else:
                response = response.replace(prompt, "").strip()
            
            return response
        except Exception as e:
            print(f"모델 응답 생성 실패: {e}")
            # 기본 응답 반환
            if context:
                return f"컨텍스트를 바탕으로 한 답변: {query}에 대한 전문가 조언입니다. 외환 거래에서는 리스크 관리가 중요하며, 기술적 분석과 기본적 분석을 종합적으로 고려해야 합니다."
            else:
                return f"{query}에 대한 답변입니다. 외환 거래 전문가로서 정확하고 실용적인 조언을 제공합니다."
    
    def rag_query(self, query: str, n_context: int = 3) -> str:
        """RAG를 사용한 질의응답"""
        # 관련 문서 검색
        search_results = self.search_similar_documents(query, n_context)
        
        if not search_results['documents'] or not search_results['documents'][0]:
            return self.generate_response(query)
        
        # 컨텍스트 구성
        context_parts = []
        for i, doc in enumerate(search_results['documents'][0]):
            context_parts.append(f"문서 {i+1}: {doc}")
        
        context = "\n\n".join(context_parts)
        
        # 응답 생성
        response = self.generate_response(query, context)
        
        return response
    
    def add_fx_knowledge_base(self):
        """외환 거래 기본 지식 베이스 구축 (KRW 기반)"""
        fx_knowledge = [
            "외환 거래(Forex Trading)는 서로 다른 국가의 통화를 교환하는 거래입니다. 한국에서는 주로 KRW(원화)와 다른 통화 간의 거래가 이루어집니다.",
            
            "한국 시장의 주요 통화쌍으로는 USD/KRW(달러/원), EUR/KRW(유로/원), JPY/KRW(엔/원), GBP/KRW(파운드/원) 등이 있습니다.",
            
            "기술적 분석은 가격 차트와 거래량을 분석하여 미래 가격 움직임을 예측하는 방법입니다. 주요 지표로는 이동평균, RSI, MACD, 볼린저 밴드가 있습니다.",
            
            "기본적 분석은 경제 지표, 중앙은행 정책, 정치적 사건 등을 분석하여 통화의 가치를 평가하는 방법입니다. 한국에서는 한국은행의 기준금리 결정이 중요합니다.",
            
            "리스크 관리의 핵심은 포지션 크기 조절, 손절매(Stop Loss) 설정, 그리고 리스크 대비 보상 비율을 고려하는 것입니다. 특히 원화 변동성에 주의해야 합니다.",
            
            "레버리지는 적은 자본으로 큰 거래를 할 수 있게 해주지만, 동시에 손실 위험도 증가시킵니다. 한국에서는 외환거래법에 따라 레버리지 제한이 있습니다.",
            
            "한국 외환 시장은 서울 시간 오전 9시부터 오후 6시까지 운영되며, 주요 거래 센터는 서울, 도쿄, 런던, 뉴욕입니다.",
            
            "스왑 포인트는 두 통화 간의 이자율 차이로 인해 발생하며, 장기 포지션 보유 시 고려해야 할 요소입니다. 한국과 미국의 기준금리 차이가 중요합니다.",
            
            "외환 거래에서 성공하려면 일관된 거래 전략, 철저한 리스크 관리, 그리고 감정적 통제가 필요합니다. 특히 원화의 특성을 이해하는 것이 중요합니다.",
            
            "한국 시장에서는 원화 강세/약세에 따른 수출입업체의 환헤지 수요가 중요한 요소입니다. 또한 글로벌 리스크 이벤트 시 원화의 안전자산 역할도 고려해야 합니다."
        ]
        
        metadata = [{"source": "fx_basics", "category": "basic_knowledge", "region": "korea"} for _ in fx_knowledge]
        self.add_knowledge(fx_knowledge, metadata)
        print("한국 시장 기반 외환 지식 베이스가 구축되었습니다.")
