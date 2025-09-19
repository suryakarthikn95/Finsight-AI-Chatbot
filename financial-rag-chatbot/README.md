FinSight.AI is an interactive financial chatbot built with Streamlit + LLMs.
It helps you quickly look up stock quotes, price trends, and now even scan entire indexes like DAX, FTSE100, CAC40, NIFTY50, NASDAQ100, S&P sample, and NIKKEI225.

🚀 Features:
	•	Chat-based interface → Ask in plain English, e.g.
	•	"Apple latest price"
	•	"trend for Reliance last 5 days"
	•	"quote META"
	•	Index & Exchange Screener → Get top gainers, losers, or most active stocks in major indexes:
	•	"top 5 companies in the DAX"
	•	"what’s moving in FTSE 100 today"
	•	"biggest losers in NASDAQ 100"
	•	Fast Data Retrieval → Uses Yahoo Finance + TD API with fallbacks & caching.
	•	Dark/Light Theme Toggle.
	•	Optional summaries → LLM-powered concise summaries of quotes and trends.

🛠 Setup
1. Clone repo:
    git clone https://github.com/your-username/finsight-ai.git
    cd finsight-ai

2.  Install requirements:
    pip install -r requirements.txt

3. Environment variables. Copy .env.example → .env and add your keys:
    OPENAI_API_KEY=your_key_here
    TD_API_KEY=your_td_key_here

4. Run the application:
    streamlit run chatbot.py

Example Queries
	•	"Infosys latest price"
	•	"trend for Tesla last week"
	•	"which companies are worth looking at in NIFTY 50 today"
	•	"top 3 losers in CAC40"
	•	"most active in NASDAQ 100"

🔮 Roadmap
	•	Portfolio-style watchlists
	•	More global indexes (Shanghai, Hang Seng, etc.)
	•	Advanced analytics (PE ratios, news sentiment)
