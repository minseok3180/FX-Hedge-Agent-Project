"""
collector.py를 6번 돌려서 2025-08-25부터 2025-08-30까지 매일 데이터 수집
"""

import datetime
import time
from collector import DataCollector

def collect_6days():
    """6일간 데이터 수집 (2025-08-25 ~ 2025-08-30)"""
    
    # 수집할 날짜들
    start_date = datetime.date(2025, 8, 25)
    end_date = datetime.date(2025, 8, 30)
    
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += datetime.timedelta(days=1)
    
    print("🚀 6일간 데이터 수집 시작")
    print("=" * 60)
    print(f"📅 수집 기간: {start_date} ~ {end_date}")
    print(f"📊 총 {len(dates)}일")
    print("=" * 60)
    
    collector = DataCollector()
    max_articles_per_keyword = 3
    
    success_count = 0
    total_news = 0
    total_time = 0
    
    for i, date in enumerate(dates, 1):
        print(f"\n🔄 [{i}/{len(dates)}] {date} 데이터 수집 시작...")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # 데이터 수집 실행
            parsed_data, saved_files = collector.collect_daily_data(date, max_articles_per_keyword)
            
            # 결과 확인
            news_count = len(parsed_data['news_articles'])
            exchange_available = bool(parsed_data['exchange_rate'])
            
            end_time = time.time()
            execution_time = end_time - start_time
            total_time += execution_time
            
            print(f"\n✅ {date} 수집 완료!")
            print(f"   ⏰ 실행시간: {execution_time:.1f}초")
            print(f"   📊 환율데이터: {'✅' if exchange_available else '❌'}")
            print(f"   📰 뉴스기사: {news_count}개")
            print(f"   💾 저장된 파일:")
            
            for key, filename in saved_files.items():
                if filename:
                    print(f"      - {key}: {filename}")
            
            success_count += 1
            total_news += news_count
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            total_time += execution_time
            
            print(f"\n❌ {date} 수집 실패!")
            print(f"   ⏰ 실행시간: {execution_time:.1f}초")
            print(f"   🚨 오류: {e}")
        
        # 다음 수집까지 대기 (마지막 날 제외)
        if i < len(dates):
            print(f"\n⏳ 다음 수집까지 5초 대기...")
            time.sleep(5)
    
    # 최종 결과 요약
    print("\n" + "=" * 60)
    print("🎉 6일간 데이터 수집 완료!")
    print("=" * 60)
    print(f"📅 수집 기간: {start_date} ~ {end_date}")
    print(f"✅ 성공: {success_count}일")
    print(f"❌ 실패: {len(dates) - success_count}일")
    print(f"📰 총 뉴스: {total_news}개")
    print(f"⏰ 총 실행시간: {total_time:.1f}초")
    print(f"📊 평균 실행시간: {total_time/len(dates):.1f}초/일")
    
    if success_count == len(dates):
        print("\n🎯 모든 날짜 수집 성공! 🎯")
    else:
        print(f"\n⚠️ {len(dates) - success_count}일 수집 실패")
    
    print("\n📁 생성된 파일들:")
    print("   - daily_data_YYYYMMDD.json (전체 데이터)")
    print("   - exchange_rate_YYYYMMDD.csv (환율 데이터)")
    print("   - news_articles_YYYYMMDD.csv (뉴스 데이터)")
    print("   - news_summary_YYYYMMDD.csv (뉴스 요약)")

if __name__ == "__main__":
    collect_6days()
