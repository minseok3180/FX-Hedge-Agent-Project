from src.config import Config
from src.data_collector import collect_fx_timeseries, collect_fx_news, save_collected
from src.data_forecasting import lstm_forecast_returns
from src.rag_parsing import build_rag_db


def main():
    cfg = Config()
    # 1) 수집
    ts = collect_fx_timeseries(cfg.fx_symbol, cfg.start_date, cfg.end_date)
    news = collect_fx_news(n=cfg.news_count)
    ts_csv, news_csv = save_collected(cfg.base_data_dir, ts, news)

    # 2) 예측 (학습 실패/미설치 시 내부 더미로 폴백됨)
    _pred_tr, _pred_te = lstm_forecast_returns(ts)

    # 3) RAG DB 구성
    build_rag_db(cfg.vector_db_dir, cfg.embedding_model, ts_csv, news_csv)

    print("수집/예측/RAG 업데이트 완료. Streamlit 실행: streamlit run src/streamlit_app.py")


if __name__ == "__main__":
    main()


