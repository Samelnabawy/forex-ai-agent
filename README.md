# Forex AI Trading Agent

Multi-agent AI system for forex day-trading signals across 9 instruments.

## Architecture

9 specialized agents that reason like an institutional trading desk:

| Agent | Role | Tech |
|-------|------|------|
| Technical Analyst | Indicators, patterns, multi-TF analysis | pandas-ta |
| Macro Analyst | Central bank policy, economic data | Claude API |
| Correlation Agent | Cross-instrument relationships | numpy/scipy |
| Sentiment Agent | News NLP, COT positioning | Claude API |
| Bull Researcher | Builds case FOR a trade | Claude API + RAG |
| Bear Researcher | Builds case AGAINST a trade | Claude API + RAG |
| Risk Manager | Enforces rules, has VETO power | Hardcoded rules |
| Portfolio Manager | Orchestrator, final decision | LangGraph |
| Historical Brain | RAG knowledge base, 20+ years | pgvector |

## Instruments

EUR/USD · GBP/USD · USD/JPY · USD/CHF · AUD/USD · USD/CAD · NZD/USD · XAU/USD (Gold) · WTI (Oil)

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env with your API keys

# 2. Start infrastructure
docker compose up -d postgres redis

# 3. Verify database
docker compose exec postgres psql -U forex -d forex_agent -c "SELECT 1;"

# 4. Backfill historical data
python -m scripts.backfill_prices --all

# 5. Start the application
docker compose up -d
```

## Stack

- **Python 3.12** — async-first with type hints
- **LangGraph** — agent orchestration
- **Claude API** — reasoning (Sonnet) + classification (Haiku)
- **PostgreSQL 16** + pgvector + TimescaleDB
- **Redis** + pub/sub — real-time data pipeline
- **FastAPI** — REST + WebSocket
- **Docker Compose** — containerized deployment

## Project Structure

```
src/
├── config/          # Settings, instruments, risk rules
├── data/            # Ingestion, storage, historical backfill
├── agents/          # All 9 agents (base_agent.py + implementations)
├── brain/           # Historical Brain RAG interface
├── execution/       # Telegram bot, trade logger
├── orchestration/   # LangGraph workflow, debate, scheduler
└── monitoring/      # FastAPI dashboard, Prometheus metrics
```

## Risk Rules (Non-Negotiable)

1. Max 1% risk per trade
2. Daily loss limit: -2%
3. Weekly loss limit: -5%
4. Max 2 correlated trades
5. News blackout: ±15 min of high-impact events
6. Spread filter: 2x average
7. London + NY sessions only
8. Mandatory stop loss
9. Circuit breaker: 3 consecutive losses → 2 hour pause
10. All positions closed Friday 20:00 UTC

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/ tests/
mypy src/
```

## Reference

See `PROJECT_SPEC.md` for the complete specification.
