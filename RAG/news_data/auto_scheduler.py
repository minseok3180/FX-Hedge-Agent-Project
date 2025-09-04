"""
Daily Data Collection Automation Scheduler
collector.py를 매일 자동으로 실행하는 스케줄러

실행 방법:
1. 백그라운드 실행: python auto_scheduler.py
2. 특정 시간 설정하여 실행
3. cron job으로 시스템 레벨에서 실행
"""

import schedule
import time
import datetime
import logging
import os
import subprocess
import json
from pathlib import Path
from collector import DataCollector

class AutoScheduler:
    def __init__(self, config_file="scheduler_config.json"):
        """자동 스케줄러 초기화"""
        self.config_file = config_file
        self.config = self.load_config()
        self.setup_logging()
        
    def load_config(self):
        """설정 파일 로드 또는 기본 설정 생성"""
        default_config = {
            "schedule_time": "09:00",  # 매일 오전 9시
            "max_articles_per_keyword": 3,
            "enabled": True,
            "weekends_enabled": True,
            "holidays_enabled": False,
            "data_directory": "./daily_data",
            "log_directory": "./logs",
            "notification": {
                "enabled": False,
                "email": "",
                "webhook_url": ""
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # 기본값과 병합
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                print(f"설정 파일 로드 실패, 기본 설정 사용: {e}")
                
        # 기본 설정 파일 생성
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        print(f"기본 설정 파일 생성: {self.config_file}")
        
        return default_config
    
    def setup_logging(self):
        """로깅 설정"""
        log_dir = Path(self.config["log_directory"])
        log_dir.mkdir(exist_ok=True)
        
        # 로그 파일명: scheduler_YYYYMMDD.log
        today = datetime.datetime.now().strftime('%Y%m%d')
        log_file = log_dir / f"scheduler_{today}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def should_run_today(self):
        """오늘 실행해야 하는지 확인"""
        if not self.config["enabled"]:
            return False
            
        today = datetime.datetime.now()
        weekday = today.weekday()  # 0=월요일, 6=일요일
        
        # 주말 체크 (5=토요일, 6=일요일)
        if weekday >= 5 and not self.config["weekends_enabled"]:
            self.logger.info("주말이므로 수집 건너뛰기")
            return False
            
        # 공휴일 체크 (간단한 예시, 실제로는 더 정교한 공휴일 API 사용 가능)
        if not self.config["holidays_enabled"]:
            # 한국의 주요 공휴일 (간단한 예시)
            holidays_2025 = [
                "2025-01-01",  # 신정
                "2025-01-28", "2025-01-29", "2025-01-30",  # 설날
                "2025-03-01",  # 삼일절
                "2025-05-05",  # 어린이날
                "2025-05-15",  # 부처님오신날
                "2025-06-06",  # 현충일
                "2025-08-15",  # 광복절
                "2025-09-06", "2025-09-07", "2025-09-08",  # 추석
                "2025-10-03",  # 개천절
                "2025-10-09",  # 한글날
                "2025-12-25",  # 크리스마스
            ]
            
            today_str = today.strftime('%Y-%m-%d')
            if today_str in holidays_2025:
                self.logger.info(f"공휴일({today_str})이므로 수집 건너뛰기")
                return False
        
        return True
    
    def collect_today_data(self):
        """오늘 데이터 수집 실행"""
        if not self.should_run_today():
            return False
            
        try:
            self.logger.info("=" * 60)
            self.logger.info("📅 일일 자동 데이터 수집 시작")
            self.logger.info("=" * 60)
            
            # 오늘 날짜
            today = datetime.date.today()
            
            # 데이터 디렉토리 생성
            data_dir = Path(self.config["data_directory"])
            data_dir.mkdir(exist_ok=True)
            
            # 기존 디렉토리로 이동 (파일 저장 위치)
            original_dir = os.getcwd()
            os.chdir(data_dir)
            
            try:
                # 데이터 수집 실행
                collector = DataCollector()
                parsed_data, saved_files = collector.collect_daily_data(
                    today, 
                    self.config["max_articles_per_keyword"]
                )
                
                # 수집 결과 로깅
                news_count = len(parsed_data['news_articles'])
                exchange_available = bool(parsed_data['exchange_rate'])
                
                self.logger.info(f"✅ {today} 데이터 수집 완료!")
                self.logger.info(f"📊 환율 데이터: {'✅' if exchange_available else '❌'}")
                self.logger.info(f"📰 뉴스 기사: {news_count}개")
                self.logger.info(f"💾 저장된 파일:")
                
                for key, filename in saved_files.items():
                    if filename:
                        self.logger.info(f"  - {key}: {filename}")
                
                # 알림 전송 (설정된 경우)
                self.send_notification(True, today, news_count, exchange_available)
                
                return True
                
            finally:
                # 원래 디렉토리로 복귀
                os.chdir(original_dir)
                
        except Exception as e:
            self.logger.error(f"❌ 데이터 수집 실패: {e}")
            self.send_notification(False, today, 0, False, str(e))
            return False
    
    def send_notification(self, success, date, news_count, exchange_available, error=None):
        """알림 전송 (이메일, 웹훅 등)"""
        if not self.config["notification"]["enabled"]:
            return
            
        status = "성공" if success else "실패"
        message = f"""
📅 일일 데이터 수집 {status}
날짜: {date}
환율 데이터: {'✅' if exchange_available else '❌'}
뉴스 기사: {news_count}개
"""
        
        if error:
            message += f"\n❌ 오류: {error}"
        
        # 웹훅 알림 (Slack, Discord 등)
        webhook_url = self.config["notification"]["webhook_url"]
        if webhook_url:
            try:
                import requests
                payload = {"text": message}
                requests.post(webhook_url, json=payload, timeout=10)
                self.logger.info("웹훅 알림 전송 완료")
            except Exception as e:
                self.logger.error(f"웹훅 알림 실패: {e}")
    
    def manual_run(self, target_date=None):
        """수동 실행 (특정 날짜)"""
        if target_date is None:
            target_date = datetime.date.today()
        
        self.logger.info(f"🔧 수동 실행: {target_date}")
        
        try:
            collector = DataCollector()
            parsed_data, saved_files = collector.collect_daily_data(
                target_date, 
                self.config["max_articles_per_keyword"]
            )
            
            self.logger.info(f"✅ {target_date} 수동 수집 완료!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {target_date} 수동 수집 실패: {e}")
            return False
    
    def start_scheduler(self):
        """스케줄러 시작"""
        schedule_time = self.config["schedule_time"]
        
        # 매일 지정된 시간에 실행
        schedule.every().day.at(schedule_time).do(self.collect_today_data)
        
        self.logger.info(f"🚀 자동 스케줄러 시작")
        self.logger.info(f"⏰ 실행 시간: 매일 {schedule_time}")
        self.logger.info(f"📂 데이터 저장: {self.config['data_directory']}")
        self.logger.info(f"📋 로그 저장: {self.config['log_directory']}")
        self.logger.info("스케줄러를 중단하려면 Ctrl+C를 누르세요")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # 1분마다 체크
                
        except KeyboardInterrupt:
            self.logger.info("🛑 스케줄러가 사용자에 의해 중단되었습니다")
        except Exception as e:
            self.logger.error(f"🚨 스케줄러 오류: {e}")

def main():
    """메인 실행 함수"""
    print("💰 Daily Data Collection Scheduler")
    print("=" * 50)
    
    scheduler = AutoScheduler()
    
    while True:
        print("\n실행 모드를 선택하세요:")
        print("1. 자동 스케줄러 시작 (백그라운드 실행)")
        print("2. 수동 실행 (오늘)")
        print("3. 수동 실행 (특정 날짜)")
        print("4. 설정 확인")
        print("5. 종료")
        
        choice = input("선택 (1-5): ").strip()
        
        if choice == "1":
            scheduler.start_scheduler()
            break
            
        elif choice == "2":
            success = scheduler.manual_run()
            if success:
                print("✅ 수동 실행 완료!")
            else:
                print("❌ 수동 실행 실패!")
                
        elif choice == "3":
            date_str = input("날짜 입력 (YYYY-MM-DD): ").strip()
            try:
                target_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
                success = scheduler.manual_run(target_date)
                if success:
                    print(f"✅ {target_date} 수동 실행 완료!")
                else:
                    print(f"❌ {target_date} 수동 실행 실패!")
            except ValueError:
                print("❌ 잘못된 날짜 형식입니다")
                
        elif choice == "4":
            print("\n📋 현재 설정:")
            for key, value in scheduler.config.items():
                print(f"  {key}: {value}")
            input("\n계속하려면 Enter를 누르세요...")
            
        elif choice == "5":
            print("👋 스케줄러를 종료합니다")
            break
            
        else:
            print("❌ 잘못된 선택입니다")

if __name__ == "__main__":
    main()
