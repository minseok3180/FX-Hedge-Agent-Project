"""
Hedge Agent Simulation Workflow - Data Collector
수집 모듈: 정형 데이터(환율) + 비정형 데이터(뉴스) 수집 및 파싱
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import datetime
import time
import re
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import json

class DataCollector:
    """정형/비정형 데이터 수집 및 파싱 클래스"""
    
    def __init__(self):
        """초기화"""
        self.driver = None
        self.exchange_keywords = [
            "환율", "매매기준율", "재정환율", "스프레드", 
            "거래위험", "경제적 노출", "통화 미스매치", "오픈 포지션", 
            "선물환", "NDF", "통화선물", "통화옵션", "스왑", "헤지", 
            "외환", "구매력평가", "금리평가", "달러인덱스", "스왑 포인트", 
            "헤지 프리미엄", "달러예금", "USD/KRW", "자본유출", 
            "미국무역", "미국관세"
        ]
        
    def contains_exchange_keyword(self, text, keywords):
        """텍스트에 환율 관련 키워드가 포함되어 있는지 확인"""
        if not text:
            return False
        text_lower = text.lower()
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return True
        return False

    def get_matched_keywords(self, text, keywords, debug=False):
        """텍스트에 포함된 환율 관련 키워드들을 반환"""
        matched = []
        if not text:
            return matched
        text_lower = text.lower()
        
        if debug:
            print(f"        디버그: 텍스트 길이 {len(text)}자에서 키워드 검색 중...")
            
        for keyword in keywords:
            if keyword.lower() in text_lower:
                matched.append(keyword)
                if debug:
                    print(f"        디버그: '{keyword}' 키워드 발견!")
        
        if debug and not matched:
            print(f"        디버그: 키워드 미발견. 텍스트 시작 부분: {text[:100]}...")
            
        return matched
    
    def extract_publish_date(self, parent_element, target_date):
        """뉴스 발행 날짜 추출 및 검증"""
        try:
            target_str = target_date.strftime('%Y-%m-%d')
            target_formats = [
                target_date.strftime('%Y.%m.%d'),
                target_date.strftime('%Y-%m-%d'),
                target_date.strftime('%m.%d'),
                target_date.strftime('%m-%d'),
                target_date.strftime('%m/%d'),
                f"{target_date.month}월{target_date.day}일"
            ]
            
            # 시간 정보 찾기 시도
            time_elements = []
            
            # 다양한 시간 관련 클래스와 태그 찾기
            time_selectors = [
                'span.info',
                'span[class*="time"]',
                'span[class*="date"]',
                'time',
                'span.sub_txt'
            ]
            
            for selector in time_selectors:
                elements = parent_element.select(selector)
                time_elements.extend(elements)
            
            # 발행 시간 텍스트 추출
            for element in time_elements:
                time_text = element.get_text().strip()
                
                # 목표 날짜 형식과 매칭 확인
                for date_format in target_formats:
                    if date_format in time_text:
                        print(f"        🕐 날짜 매칭: {time_text}")
                        return time_text
                
                # "X시간 전", "X분 전" 등의 상대 시간 체크
                if any(word in time_text for word in ['시간 전', '분 전', '방금 전']):
                    print(f"        🕐 상대 시간: {time_text}")
                    return time_text
            
            # URL에서 날짜 정보 추출 시도
            try:
                link = parent_element.find('a', href=lambda x: x and 'news.naver.com' in x)
                if link:
                    url = link.get('href')
                    # URL에서 날짜 패턴 찾기
                    import re
                    date_pattern = r'(\d{4})(\d{2})(\d{2})'
                    match = re.search(date_pattern, url)
                    if match:
                        url_date = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                        if url_date == target_str:
                            print(f"        🔗 URL 날짜 매칭: {url_date}")
                            return url_date
            except:
                pass
            
            return None
            
        except Exception as e:
            print(f"        ❌ 날짜 추출 실패: {e}")
            return None
    
    def get_news_with_selenium(self, search_url, target_date):
        """Selenium으로 뉴스 검색 (백업 방법)"""
        try:
            if not self.setup_driver():
                return []
            
            self.driver.get(search_url)
            time.sleep(3)
            
            # 네이버의 동적 로딩 대기
            headlines = self.driver.find_elements(By.CSS_SELECTOR, 'span[class*="headline"]')
            
            # BeautifulSoup 객체로 변환하여 기존 로직과 호환
            selenium_headlines = []
            for element in headlines:
                try:
                    # Selenium element를 HTML로 변환
                    html = element.get_attribute('outerHTML')
                    soup_element = BeautifulSoup(html, 'html.parser').find('span')
                    if soup_element:
                        selenium_headlines.append(soup_element)
                except:
                    continue
            
            return selenium_headlines
            
        except Exception as e:
            print(f"    Selenium 백업 실패: {e}")
            return []
        finally:
            if self.driver:
                try:
                    self.driver.quit()
                    self.driver = None
                except:
                    pass
        
    def setup_driver(self):
        """Chrome WebDriver 설정"""
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            
            chrome_options = Options()
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            return True
            
        except Exception as e:
            print(f"Chrome WebDriver 초기화 실패: {e}")
            return False
    
    def collect_exchange_rate_data(self, target_date):
        """정형 데이터: yfinance로 환율 데이터 수집"""
        print(f"📊 {target_date} 환율 데이터 수집 중...")
        
        try:
            # USD/KRW 환율 데이터
            usd_krw = yf.Ticker("KRW=X")
            
            # 해당 날짜 전후 데이터 가져오기 (주말/공휴일 대비)
            start_date = target_date - datetime.timedelta(days=5)
            end_date = target_date + datetime.timedelta(days=2)
            
            hist_data = usd_krw.history(start=start_date, end=end_date)
            
            if not hist_data.empty:
                # 가장 가까운 날짜 데이터 찾기
                target_str = target_date.strftime('%Y-%m-%d')
                if target_str in hist_data.index.strftime('%Y-%m-%d').tolist():
                    day_data = hist_data[hist_data.index.strftime('%Y-%m-%d') == target_str].iloc[0]
                else:
                    # 가장 가까운 이전 날짜 데이터 사용
                    day_data = hist_data.iloc[-1]
                
                exchange_data = {
                    'date': target_date.strftime('%Y-%m-%d'),
                    'usd_krw_open': round(day_data['Open'], 2),
                    'usd_krw_high': round(day_data['High'], 2),
                    'usd_krw_low': round(day_data['Low'], 2),
                    'usd_krw_close': round(day_data['Close'], 2),
                    'volume': int(day_data['Volume']) if not pd.isna(day_data['Volume']) else 0,
                    'collected_at': datetime.datetime.now().isoformat()
                }
                
                print(f"✅ 환율 데이터 수집 완료: USD/KRW {exchange_data['usd_krw_close']}")
                return exchange_data
            else:
                print("❌ 환율 데이터를 찾을 수 없습니다.")
                return None
                
        except Exception as e:
            print(f"❌ 환율 데이터 수집 실패: {e}")
            return None
    
    def get_naver_news_full_content(self, news_url):
        """네이버 뉴스 본문 내용 추출"""
        try:
            # 네이버 뉴스 URL인지 확인
            if 'news.naver.com' not in news_url:
                return None
            
            # 기사 본문 페이지 요청
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.get(news_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 네이버 뉴스 본문 추출 (여러 선택자 시도)
                content = ""
                
                # 방법 1: newsct_article 클래스
                article_body = soup.find('div', class_='newsct_article')
                if article_body:
                    # 불필요한 요소 제거
                    for unwanted in article_body.find_all(['script', 'style', 'iframe', 'ad']):
                        unwanted.decompose()
                    content = article_body.get_text(strip=True)
                
                # 방법 2: go_trans._article_content
                if not content:
                    article_body = soup.find('div', {'id': 'newsct_article'})
                    if article_body:
                        content = article_body.get_text(strip=True)
                
                # 방법 3: article 태그
                if not content:
                    article_body = soup.find('article')
                    if article_body:
                        content = article_body.get_text(strip=True)
                
                # 방법 4: 일반적인 본문 패턴
                if not content:
                    article_body = soup.find('div', class_=re.compile(r'article|content|body'))
                    if article_body:
                        content = article_body.get_text(strip=True)
                
                # 댓글 수 추출
                comment_count = 0
                try:
                    # 댓글 수 패턴 찾기
                    comment_element = soup.find('span', class_=re.compile(r'comment|reply'))
                    if comment_element:
                        comment_text = comment_element.get_text()
                        comment_numbers = re.findall(r'\d+', comment_text)
                        if comment_numbers:
                            comment_count = int(comment_numbers[0])
                except:
                    pass
                
                return {
                    'content': content[:2000] if content else "",  # 본문 2000자 제한
                    'comment_count': comment_count
                }
            else:
                return None
                
        except Exception as e:
            print(f"본문 추출 실패 ({news_url}): {e}")
            return None
    
    def collect_naver_news_full(self, target_date, max_articles_per_keyword=5):
        """비정형 데이터: 네이버 뉴스 전체 내용(헤드라인, 본문, URL, 댓글수) 수집"""
        print(f"📰 {target_date} 네이버 뉴스 전체 내용 수집 중...")
        
        all_news = []
        ds = target_date.strftime('%Y.%m.%d')
        
        # requests로 먼저 시도 (더 안정적) - 날짜별 차별화를 위해 헤더 개선
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
        
        try:
            for keyword_idx, keyword in enumerate(self.exchange_keywords):
                print(f"🔍 [{keyword_idx+1}/{len(self.exchange_keywords)}] '{keyword}' 검색 중...")
                
                # 네이버 뉴스 검색 URL (날짜별 차별화가 확실한 방법)
                ds_formatted = ds.replace('.', '')  # 20250830 형태
                search_url = f"https://search.naver.com/search.naver?where=news&query={keyword}&ds={ds}&de={ds}&nso=so%3Ar%2Ca%3Aall%2Cp%3Afrom{ds_formatted}to{ds_formatted}"
                print(f"    검색 URL: {search_url}")
                
                try:
                    response = requests.get(search_url, headers=headers, timeout=10)
                    if response.status_code != 200:
                        print(f"    ❌ 요청 실패: {response.status_code}")
                        continue
                        
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # 뉴스 제목 추출
                    news_items = []
                    
                    # 방법 1: headline 클래스 span 요소들
                    headlines = soup.find_all('span', class_=lambda x: x and 'headline' in str(x))
                    print(f"    headline 요소: {len(headlines)}개 발견")
                    
                    # 실제 뉴스가 없으면 Selenium으로 재시도
                    if len(headlines) == 0:
                        print(f"    requests 실패, Selenium으로 재시도...")
                        headlines = self.get_news_with_selenium(search_url, target_date)
                        print(f"    Selenium 결과: {len(headlines)}개 발견")
                    
                    for headline in headlines[:max_articles_per_keyword]:
                        title = headline.get_text().strip()
                        if title and len(title) > 10 and len(title) < 200:
                            # 해당 제목과 연결된 링크 찾기
                            try:
                                # 부모 컨테이너에서 뉴스 링크 찾기
                                parent = headline.find_parent()
                                while parent and not parent.find('a', href=lambda x: x and 'news.naver.com' in x):
                                    parent = parent.find_parent()
                                    if not parent or parent.name == 'html':
                                        break
                                
                                if parent:
                                    link = parent.find('a', href=lambda x: x and 'news.naver.com' in x)
                                    if link:
                                        url = link.get('href')
                                        
                                        # 발행 날짜 확인 (간소화된 검증)
                                        publish_date = self.extract_publish_date(parent, target_date)
                                        if not publish_date:
                                            # 날짜를 찾지 못해도 포함 (nso 파라미터로 이미 날짜 필터링됨)
                                            publish_date = target_date.strftime('%Y-%m-%d')
                                        
                                        # 키워드 확인
                                        title_keywords = self.get_matched_keywords(title, self.exchange_keywords)
                                        
                                        news_items.append({
                                            'title': title,
                                            'url': url,
                                            'title_keywords': title_keywords,
                                            'publish_time': publish_date
                                        })
                                        
                                        if title_keywords:
                                            print(f"      ✅ 제목 키워드 매칭: {title[:50]}... (키워드: {', '.join(title_keywords)})")
                                        else:
                                            print(f"      📝 제목 키워드 없음, 본문 확인 예정: {title[:50]}...")
                                            
                            except Exception as e:
                                print(f"      링크 찾기 실패: {e}")
                                continue
                    
                    print(f"    키워드 '{keyword}': {len(news_items)}개 뉴스 발견")
                    
                except Exception as e:
                    print(f"    검색 요청 실패: {e}")
                    continue
                
                # 각 뉴스 기사의 본문 내용 수집
                keyword_news = []
                for news_item in news_items:
                    try:
                        print(f"    📄 본문 수집 중: {news_item['title'][:50]}...")
                        
                        # 본문 내용 가져오기
                        full_content = self.get_naver_news_full_content(news_item['url'])
                        
                        if full_content and full_content['content']:
                            # 제목과 본문에서 키워드 확인
                            title_keywords = news_item.get('title_keywords', [])
                            content_keywords = self.get_matched_keywords(full_content['content'], self.exchange_keywords, debug=True)
                            all_matched_keywords = list(set(title_keywords + content_keywords))
                            
                            print(f"      🔍 키워드 분석:")
                            print(f"         제목 키워드: {title_keywords if title_keywords else '없음'}")
                            print(f"         본문 키워드: {content_keywords if content_keywords else '없음'}")
                            print(f"         전체 매칭: {all_matched_keywords if all_matched_keywords else '없음'}")
                            
                            # 제목이나 본문에 키워드가 있으면 포함
                            if all_matched_keywords:
                                news_data = {
                                    'date': target_date.strftime('%Y-%m-%d'),
                                    'search_keyword': keyword,
                                    'title': news_item['title'],
                                    'content': full_content['content'],
                                    'url': news_item['url'],
                                    'publish_time': news_item['publish_time'],
                                    'comment_count': full_content['comment_count'],
                                    'matched_keywords': ', '.join(all_matched_keywords),
                                    'title_matched': ', '.join(title_keywords),
                                    'content_matched': ', '.join(content_keywords),
                                    'collected_at': datetime.datetime.now().isoformat()
                                }
                                
                                keyword_news.append(news_data)
                                print(f"      ✅ 수집 완료 (본문 {len(full_content['content'])}자, 댓글 {full_content['comment_count']}개)")
                                print(f"         최종 매칭 키워드: {', '.join(all_matched_keywords)}")
                            else:
                                print(f"      ❌ 제목과 본문 모두에서 키워드 미발견")
                        else:
                            print(f"      ❌ 본문 수집 실패")
                        
                        time.sleep(2)  # 요청 간격
                        
                    except Exception as e:
                        print(f"      ❌ 오류: {e}")
                        continue
                
                all_news.extend(keyword_news)
                print(f"  ✅ '{keyword}': {len(keyword_news)}개 기사 수집 완료")
                
                time.sleep(3)  # 키워드 간 대기
            
        except Exception as e:
            print(f"❌ 뉴스 수집 중 오류: {e}")
        
        print(f"📰 총 {len(all_news)}개 뉴스 기사 수집 완료")
        return all_news
    
    def parse_and_format_data(self, exchange_data, news_data, target_date):
        """수집된 데이터 파싱 및 포맷팅"""
        print(f"🔄 {target_date} 데이터 파싱 및 포맷팅 중...")
        
        parsed_data = {
            'collection_date': target_date.strftime('%Y-%m-%d'),
            'collection_timestamp': datetime.datetime.now().isoformat(),
            'exchange_rate': exchange_data if exchange_data else {},
            'news_articles': news_data if news_data else [],
            'summary': {
                'total_news_articles': len(news_data) if news_data else 0,
                'keywords_with_news': len(set([article['search_keyword'] for article in news_data])) if news_data else 0,
                'exchange_rate_available': bool(exchange_data)
            }
        }
        
        # 키워드별 뉴스 수 통계
        if news_data:
            keyword_stats = {}
            for article in news_data:
                keyword = article['search_keyword']
                keyword_stats[keyword] = keyword_stats.get(keyword, 0) + 1
            parsed_data['summary']['keyword_stats'] = keyword_stats
        
        print(f"✅ 데이터 파싱 완료:")
        print(f"  - 환율 데이터: {'✅' if exchange_data else '❌'}")
        print(f"  - 뉴스 기사: {len(news_data)}개")
        print(f"  - 수집 키워드: {len(set([article['search_keyword'] for article in news_data]) if news_data else [])}개")
        
        return parsed_data
    
    def save_daily_data(self, parsed_data, target_date):
        """일일 수집 데이터 저장"""
        date_str = target_date.strftime('%Y%m%d')
        
        # 1. JSON 형태로 전체 데이터 저장
        json_filename = f"daily_data_{date_str}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=2)
        print(f"💾 전체 데이터 저장: {json_filename}")
        
        # 2. 환율 데이터만 CSV로 저장
        if parsed_data['exchange_rate']:
            exchange_df = pd.DataFrame([parsed_data['exchange_rate']])
            exchange_csv = f"exchange_rate_{date_str}.csv"
            exchange_df.to_csv(exchange_csv, index=False, encoding='utf-8-sig')
            print(f"💾 환율 데이터 저장: {exchange_csv}")
        
        # 3. 뉴스 데이터만 CSV로 저장
        if parsed_data['news_articles']:
            news_df = pd.DataFrame(parsed_data['news_articles'])
            news_csv = f"news_articles_{date_str}.csv"
            news_df.to_csv(news_csv, index=False, encoding='utf-8-sig')
            print(f"💾 뉴스 데이터 저장: {news_csv}")
            
            # 4. 뉴스 요약 통계 저장
            summary_data = {
                'date': target_date.strftime('%Y-%m-%d'),
                'total_articles': len(parsed_data['news_articles']),
                'keywords_count': len(set([article['search_keyword'] for article in parsed_data['news_articles']])),
                'avg_content_length': sum([len(article['content']) for article in parsed_data['news_articles']]) / len(parsed_data['news_articles']),
                'total_comments': sum([article['comment_count'] for article in parsed_data['news_articles']])
            }
            summary_df = pd.DataFrame([summary_data])
            summary_csv = f"news_summary_{date_str}.csv"
            summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
            print(f"💾 뉴스 요약 저장: {summary_csv}")
        
        return {
            'json_file': json_filename,
            'exchange_csv': f"exchange_rate_{date_str}.csv" if parsed_data['exchange_rate'] else None,
            'news_csv': f"news_articles_{date_str}.csv" if parsed_data['news_articles'] else None,
            'summary_csv': f"news_summary_{date_str}.csv" if parsed_data['news_articles'] else None
        }
    
    def collect_daily_data(self, target_date, max_articles_per_keyword=5):
        """하루치 데이터 전체 수집 프로세스"""
        print(f"🚀 {target_date} 일일 데이터 수집 시작")
        print("=" * 60)
        
        start_time = time.time()
        
        # 1. 정형 데이터 수집 (환율)
        exchange_data = self.collect_exchange_rate_data(target_date)
        
        # 2. 비정형 데이터 수집 (뉴스)
        news_data = self.collect_naver_news_full(target_date, max_articles_per_keyword)
        
        # 3. 데이터 파싱 및 포맷팅
        parsed_data = self.parse_and_format_data(exchange_data, news_data, target_date)
        
        # 4. 데이터 저장
        saved_files = self.save_daily_data(parsed_data, target_date)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print("\n" + "=" * 60)
        print(f"🎉 {target_date} 데이터 수집 완료!")
        print(f"⏰ 실행 시간: {execution_time:.1f}초")
        print(f"📁 저장된 파일들:")
        for key, filename in saved_files.items():
            if filename:
                print(f"  - {key}: {filename}")
        
        return parsed_data, saved_files


def test_news_collection():
    """뉴스 수집 테스트 함수 (소수 키워드로 빠른 테스트)"""
    print("🧪 뉴스 수집 테스트 모드")
    print("=" * 40)
    
    collector = DataCollector()
    
    # 테스트용 주요 키워드만 사용
    test_keywords = ["환율", "USD/KRW", "달러"]
    original_keywords = collector.exchange_keywords
    collector.exchange_keywords = test_keywords
    
    target_date = datetime.date(2025, 12, 30)
    
    print(f"📅 테스트 날짜: {target_date}")
    print(f"🔍 테스트 키워드: {test_keywords}")
    
    # 뉴스 수집만 테스트
    news_data = collector.collect_naver_news_full(target_date, max_articles_per_keyword=2)
    
    if news_data:
        print(f"\n✅ 테스트 성공! {len(news_data)}개 뉴스 수집")
        for i, news in enumerate(news_data[:3]):  # 처음 3개만 출력
            print(f"{i+1}. {news['title']}")
            print(f"   키워드: {news['matched_keywords']}")
            print(f"   본문 길이: {len(news['content'])}자")
    else:
        print("❌ 테스트 실패: 뉴스 수집되지 않음")
    
    # 원래 키워드 복원
    collector.exchange_keywords = original_keywords
    
    return len(news_data) if news_data else 0


def main():
    """메인 실행 함수"""
    print("💰 Hedge Agent Simulation - Data Collector")
    print("정형 데이터(환율) + 비정형 데이터(뉴스) 수집 시스템")
    print("=" * 60)
    
    # 먼저 테스트 옵션 제공
    mode = input("실행 모드를 선택하세요:\n1. 테스트 모드 (빠른 테스트)\n2. 전체 수집 모드\n선택 (1/2): ").strip()
    
    if mode == "1":
        test_count = test_news_collection()
        if test_count > 0:
            print(f"\n🎉 테스트 성공! 이제 전체 수집을 진행할 수 있습니다.")
        return
    
    # 2025년 8월 30일 설정 (실제 뉴스가 있는 날짜로 테스트)
    target_date = datetime.date(2025, 8, 30)
    max_articles_per_keyword = 3  # 키워드당 최대 기사 수
    
    print(f"📅 수집 날짜: {target_date}")
    print(f"🔍 환율 키워드: 25개")
    print(f"📰 키워드당 최대 기사 수: {max_articles_per_keyword}개")
    print(f"📊 예상 최대 뉴스 기사 수: {25 * max_articles_per_keyword}개")
    
    # 사용자 확인
    confirm = input(f"\n{target_date} 데이터 수집을 시작하시겠습니까? (y/n): ").lower().strip()
    if confirm != 'y':
        print("수집이 취소되었습니다.")
        return
    
    # 데이터 수집 실행
    collector = DataCollector()
    
    try:
        parsed_data, saved_files = collector.collect_daily_data(target_date, max_articles_per_keyword)
        
        print(f"\n✅ 모든 작업이 성공적으로 완료되었습니다!")
        print(f"🎯 다음 단계: RAG.py에서 수집된 데이터로 DB 업데이트")
        
    except Exception as e:
        print(f"\n❌ 수집 중 오류 발생: {e}")
        print("다시 시도하거나 설정을 확인해주세요.")


if __name__ == "__main__":
    main()
