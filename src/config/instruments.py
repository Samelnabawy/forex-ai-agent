"""
Instrument universe — the 9 instruments we trade.
Every property here drives pip calculation, spread filtering,
correlation grouping, and session rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum


class InstrumentCategory(StrEnum):
    MAJOR_FOREX = "major_forex"
    GOLD = "gold"
    OIL = "oil"


class CorrelationCluster(StrEnum):
    """Correlation groups from Section 2.2 — Correlation Agent spec."""
    USD_STRENGTH = "usd_strength"      # EUR/USD, GBP/USD move together
    SAFE_HAVEN = "safe_haven"          # USD/JPY, USD/CHF, Gold (risk-off)
    COMMODITY = "commodity"            # USD/CAD↔Oil, AUD/USD↔Gold
    ANTIPODEAN = "antipodean"          # AUD/USD, NZD/USD move together


@dataclass(frozen=True, slots=True)
class Instrument:
    symbol: str
    display_name: str
    category: InstrumentCategory
    pip_size: Decimal               # what 1 pip equals in price
    pip_value_per_lot: Decimal      # USD value of 1 pip on 1 standard lot
    avg_spread_pips: Decimal        # normal conditions
    avg_daily_range_pips: Decimal   # ATR benchmark
    correlation_clusters: tuple[CorrelationCluster, ...]
    base_currency: str
    quote_currency: str
    # Data source symbol overrides (some feeds use different symbols)
    polygon_symbol: str = ""
    twelve_data_symbol: str = ""


# ────────────────────────────────────────────────────────────────
# The Universe
# ────────────────────────────────────────────────────────────────

INSTRUMENTS: dict[str, Instrument] = {
    "EUR/USD": Instrument(
        symbol="EUR/USD",
        display_name="Euro / US Dollar",
        category=InstrumentCategory.MAJOR_FOREX,
        pip_size=Decimal("0.0001"),
        pip_value_per_lot=Decimal("10.00"),
        avg_spread_pips=Decimal("0.7"),
        avg_daily_range_pips=Decimal("80"),
        correlation_clusters=(CorrelationCluster.USD_STRENGTH,),
        base_currency="EUR",
        quote_currency="USD",
        polygon_symbol="C:EURUSD",
        twelve_data_symbol="EUR/USD",
    ),
    "GBP/USD": Instrument(
        symbol="GBP/USD",
        display_name="British Pound / US Dollar",
        category=InstrumentCategory.MAJOR_FOREX,
        pip_size=Decimal("0.0001"),
        pip_value_per_lot=Decimal("10.00"),
        avg_spread_pips=Decimal("1.0"),
        avg_daily_range_pips=Decimal("100"),
        correlation_clusters=(CorrelationCluster.USD_STRENGTH,),
        base_currency="GBP",
        quote_currency="USD",
        polygon_symbol="C:GBPUSD",
        twelve_data_symbol="GBP/USD",
    ),
    "USD/JPY": Instrument(
        symbol="USD/JPY",
        display_name="US Dollar / Japanese Yen",
        category=InstrumentCategory.MAJOR_FOREX,
        pip_size=Decimal("0.01"),
        pip_value_per_lot=Decimal("6.50"),  # approximate, varies with rate
        avg_spread_pips=Decimal("0.8"),
        avg_daily_range_pips=Decimal("90"),
        correlation_clusters=(CorrelationCluster.SAFE_HAVEN,),
        base_currency="USD",
        quote_currency="JPY",
        polygon_symbol="C:USDJPY",
        twelve_data_symbol="USD/JPY",
    ),
    "USD/CHF": Instrument(
        symbol="USD/CHF",
        display_name="US Dollar / Swiss Franc",
        category=InstrumentCategory.MAJOR_FOREX,
        pip_size=Decimal("0.0001"),
        pip_value_per_lot=Decimal("10.00"),
        avg_spread_pips=Decimal("1.2"),
        avg_daily_range_pips=Decimal("65"),
        correlation_clusters=(CorrelationCluster.SAFE_HAVEN,),
        base_currency="USD",
        quote_currency="CHF",
        polygon_symbol="C:USDCHF",
        twelve_data_symbol="USD/CHF",
    ),
    "AUD/USD": Instrument(
        symbol="AUD/USD",
        display_name="Australian Dollar / US Dollar",
        category=InstrumentCategory.MAJOR_FOREX,
        pip_size=Decimal("0.0001"),
        pip_value_per_lot=Decimal("10.00"),
        avg_spread_pips=Decimal("0.9"),
        avg_daily_range_pips=Decimal("70"),
        correlation_clusters=(CorrelationCluster.COMMODITY, CorrelationCluster.ANTIPODEAN),
        base_currency="AUD",
        quote_currency="USD",
        polygon_symbol="C:AUDUSD",
        twelve_data_symbol="AUD/USD",
    ),
    "USD/CAD": Instrument(
        symbol="USD/CAD",
        display_name="US Dollar / Canadian Dollar",
        category=InstrumentCategory.MAJOR_FOREX,
        pip_size=Decimal("0.0001"),
        pip_value_per_lot=Decimal("7.50"),  # approximate
        avg_spread_pips=Decimal("1.2"),
        avg_daily_range_pips=Decimal("75"),
        correlation_clusters=(CorrelationCluster.COMMODITY,),
        base_currency="USD",
        quote_currency="CAD",
        polygon_symbol="C:USDCAD",
        twelve_data_symbol="USD/CAD",
    ),
    "NZD/USD": Instrument(
        symbol="NZD/USD",
        display_name="New Zealand Dollar / US Dollar",
        category=InstrumentCategory.MAJOR_FOREX,
        pip_size=Decimal("0.0001"),
        pip_value_per_lot=Decimal("10.00"),
        avg_spread_pips=Decimal("1.3"),
        avg_daily_range_pips=Decimal("65"),
        correlation_clusters=(CorrelationCluster.COMMODITY, CorrelationCluster.ANTIPODEAN),
        base_currency="NZD",
        quote_currency="USD",
        polygon_symbol="C:NZDUSD",
        twelve_data_symbol="NZD/USD",
    ),
    "XAU/USD": Instrument(
        symbol="XAU/USD",
        display_name="Gold / US Dollar",
        category=InstrumentCategory.GOLD,
        pip_size=Decimal("0.01"),
        pip_value_per_lot=Decimal("1.00"),  # per 1 oz
        avg_spread_pips=Decimal("20"),      # in cents
        avg_daily_range_pips=Decimal("2000"),  # ~$20 range in cents
        correlation_clusters=(CorrelationCluster.SAFE_HAVEN, CorrelationCluster.COMMODITY),
        base_currency="XAU",
        quote_currency="USD",
        polygon_symbol="C:XAUUSD",
        twelve_data_symbol="XAU/USD",
    ),
    "WTI": Instrument(
        symbol="WTI",
        display_name="WTI Crude Oil",
        category=InstrumentCategory.OIL,
        pip_size=Decimal("0.01"),
        pip_value_per_lot=Decimal("10.00"),  # per barrel
        avg_spread_pips=Decimal("3"),
        avg_daily_range_pips=Decimal("200"),
        correlation_clusters=(CorrelationCluster.COMMODITY,),
        base_currency="WTI",
        quote_currency="USD",
        polygon_symbol="C:WTIUSD",
        twelve_data_symbol="USOIL",
    ),
}

# ── Convenience ─────────────────────────────────────────────
ALL_SYMBOLS: list[str] = list(INSTRUMENTS.keys())
FOREX_SYMBOLS: list[str] = [s for s, i in INSTRUMENTS.items() if i.category == InstrumentCategory.MAJOR_FOREX]
COMMODITY_SYMBOLS: list[str] = [s for s, i in INSTRUMENTS.items() if i.category in (InstrumentCategory.GOLD, InstrumentCategory.OIL)]

# Timeframes monitored (from spec Section 1.3)
EXECUTION_TIMEFRAMES: list[str] = ["5m", "15m", "1h"]
CONTEXT_TIMEFRAMES: list[str] = ["4h", "1d"]
ALL_TIMEFRAMES: list[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]

# Correlation pairs — used by the Correlation Agent for quick lookups
# Maps cluster → list of symbols in that cluster
CLUSTER_MEMBERS: dict[CorrelationCluster, list[str]] = {}
for _sym, _inst in INSTRUMENTS.items():
    for _cluster in _inst.correlation_clusters:
        CLUSTER_MEMBERS.setdefault(_cluster, []).append(_sym)
