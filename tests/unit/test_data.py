"""
Unit tests for candle aggregation logic.
Tests the pure computation — no database or Redis dependency.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.data.ingestion.candle_aggregator import (
    CandleBuilder,
    align_to_period,
    TIMEFRAME_SECONDS,
)


class TestAlignToPeriod:
    """Test period boundary alignment."""

    def test_align_5m(self) -> None:
        ts = datetime(2026, 4, 1, 14, 37, 23, tzinfo=timezone.utc)
        aligned = align_to_period(ts, TIMEFRAME_SECONDS["5m"])
        assert aligned == datetime(2026, 4, 1, 14, 35, 0, tzinfo=timezone.utc)

    def test_align_1h(self) -> None:
        ts = datetime(2026, 4, 1, 14, 37, 23, tzinfo=timezone.utc)
        aligned = align_to_period(ts, TIMEFRAME_SECONDS["1h"])
        assert aligned == datetime(2026, 4, 1, 14, 0, 0, tzinfo=timezone.utc)

    def test_align_4h(self) -> None:
        ts = datetime(2026, 4, 1, 14, 37, 23, tzinfo=timezone.utc)
        aligned = align_to_period(ts, TIMEFRAME_SECONDS["4h"])
        assert aligned == datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_align_exact_boundary(self) -> None:
        ts = datetime(2026, 4, 1, 14, 0, 0, tzinfo=timezone.utc)
        aligned = align_to_period(ts, TIMEFRAME_SECONDS["1h"])
        assert aligned == ts

    def test_align_1m(self) -> None:
        ts = datetime(2026, 4, 1, 14, 37, 23, tzinfo=timezone.utc)
        aligned = align_to_period(ts, TIMEFRAME_SECONDS["1m"])
        assert aligned == datetime(2026, 4, 1, 14, 37, 0, tzinfo=timezone.utc)


class TestCandleBuilder:
    """Test OHLCV candle construction from ticks."""

    def test_first_tick_sets_all(self) -> None:
        builder = CandleBuilder(
            "EUR/USD", "5m",
            datetime(2026, 4, 1, 14, 35, 0, tzinfo=timezone.utc),
        )
        assert builder.is_empty

        builder.update(Decimal("1.0845"))
        assert not builder.is_empty
        assert builder.open == Decimal("1.0845")
        assert builder.high == Decimal("1.0845")
        assert builder.low == Decimal("1.0845")
        assert builder.close == Decimal("1.0845")
        assert builder.tick_count == 1

    def test_multiple_ticks_ohlc(self) -> None:
        builder = CandleBuilder(
            "EUR/USD", "5m",
            datetime(2026, 4, 1, 14, 35, 0, tzinfo=timezone.utc),
        )
        builder.update(Decimal("1.0845"))  # open
        builder.update(Decimal("1.0860"))  # new high
        builder.update(Decimal("1.0830"))  # new low
        builder.update(Decimal("1.0850"))  # close

        assert builder.open == Decimal("1.0845")
        assert builder.high == Decimal("1.0860")
        assert builder.low == Decimal("1.0830")
        assert builder.close == Decimal("1.0850")
        assert builder.tick_count == 4

    def test_volume_accumulation(self) -> None:
        builder = CandleBuilder(
            "EUR/USD", "1h",
            datetime(2026, 4, 1, 14, 0, 0, tzinfo=timezone.utc),
        )
        builder.update(Decimal("1.0845"), volume=Decimal("100"))
        builder.update(Decimal("1.0850"), volume=Decimal("200"))
        builder.update(Decimal("1.0840"), volume=Decimal("150"))

        assert builder.volume == Decimal("450")

    def test_to_record(self) -> None:
        period_start = datetime(2026, 4, 1, 14, 35, 0, tzinfo=timezone.utc)
        builder = CandleBuilder("GBP/USD", "5m", period_start)
        builder.update(Decimal("1.2650"))
        builder.update(Decimal("1.2670"))
        builder.update(Decimal("1.2640"))
        builder.update(Decimal("1.2660"))

        record = builder.to_record()
        assert record.instrument == "GBP/USD"
        assert record.timeframe == "5m"
        assert record.ts == period_start
        assert record.open == Decimal("1.2650")
        assert record.high == Decimal("1.2670")
        assert record.low == Decimal("1.2640")
        assert record.close == Decimal("1.2660")

    def test_empty_candle_raises_on_to_record(self) -> None:
        builder = CandleBuilder(
            "EUR/USD", "5m",
            datetime(2026, 4, 1, 14, 35, 0, tzinfo=timezone.utc),
        )
        with pytest.raises(AssertionError):
            builder.to_record()

    def test_single_tick_candle(self) -> None:
        """Edge case: candle with only 1 tick (O=H=L=C)."""
        builder = CandleBuilder(
            "USD/JPY", "1m",
            datetime(2026, 4, 1, 14, 37, 0, tzinfo=timezone.utc),
        )
        builder.update(Decimal("150.25"))

        assert builder.open == builder.high == builder.low == builder.close
        assert builder.tick_count == 1


class TestCurrencyClassification:
    """Test headline → currency mapping."""

    def test_fed_maps_to_usd(self) -> None:
        from src.data.ingestion.news_feed import classify_currencies
        result = classify_currencies("Federal Reserve signals potential rate cut in June")
        assert "USD" in result

    def test_ecb_maps_to_eur(self) -> None:
        from src.data.ingestion.news_feed import classify_currencies
        result = classify_currencies("ECB maintains hawkish stance on inflation")
        assert "EUR" in result

    def test_oil_maps_to_wti(self) -> None:
        from src.data.ingestion.news_feed import classify_currencies
        result = classify_currencies("OPEC announces production cuts amid oil demand concerns")
        assert "WTI" in result

    def test_gold_maps_to_xau(self) -> None:
        from src.data.ingestion.news_feed import classify_currencies
        result = classify_currencies("Gold prices surge on safe haven demand")
        assert "XAU" in result

    def test_multi_currency_headline(self) -> None:
        from src.data.ingestion.news_feed import classify_currencies
        result = classify_currencies("Fed and ECB diverge on rate path, dollar weakens")
        assert "USD" in result
        assert "EUR" in result

    def test_irrelevant_headline(self) -> None:
        from src.data.ingestion.news_feed import classify_currencies
        result = classify_currencies("Tech stocks rally on AI optimism")
        assert result == []
