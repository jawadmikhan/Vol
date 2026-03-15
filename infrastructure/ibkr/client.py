"""
IBKR API client — manages the connection to TWS/Gateway and provides
methods to fetch the market data the vol strategies require.

Uses ib_insync for a clean async-friendly interface over the raw TWS API.
"""

import logging
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from ib_insync import IB, util, MarketOrder, Ticker

from .contracts import (
    UNIVERSE,
    constituent_symbols,
    spx_option,
    equity_option,
    vix_future,
    symbol_sector,
    symbol_weight,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strike-delta mapping helpers
# ---------------------------------------------------------------------------

DELTA_LABELS = ["10D_PUT", "25D_PUT", "ATM", "25D_CALL", "10D_CALL"]
TENOR_MONTHS = [1, 2, 3, 6, 9, 12]


def _nearest_expiry(expirations: list[str], target_months: int) -> str | None:
    """Pick the expiry closest to *target_months* from today."""
    today = datetime.now().date()
    target_date = today + timedelta(days=target_months * 30)
    best, best_diff = None, 9999
    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, "%Y%m%d").date()
        diff = abs((exp_date - target_date).days)
        if diff < best_diff:
            best, best_diff = exp_str, diff
    return best


def _classify_delta(delta: float | None, right: str) -> str | None:
    """Map a Greek delta to our standard delta bucket labels."""
    if delta is None:
        return None
    d = abs(delta)
    if right == "P":
        if d <= 0.15:
            return "10D_PUT"
        if d <= 0.35:
            return "25D_PUT"
        if d <= 0.60:
            return "ATM"
        return None
    else:  # Call
        if d >= 0.40:
            return "ATM"
        if d >= 0.18:
            return "25D_CALL"
        if d >= 0.05:
            return "10D_CALL"
        return None


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class IBKRClient:
    """High-level wrapper around ib_insync for vol portfolio data needs."""

    def __init__(self, host: str = "127.0.0.1", port: int = 4002, client_id: int = 1):
        """
        Args:
            host: TWS/Gateway host (default localhost).
            port: 4001=live Gateway, 4002=paper Gateway, 7496=live TWS, 7497=paper TWS.
            client_id: Unique client ID for this connection.
        """
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self):
        """Connect to TWS / IB Gateway."""
        self.ib.connect(self.host, self.port, clientId=self.client_id)
        logger.info("Connected to IBKR at %s:%s (client %s)", self.host, self.port, self.client_id)

    def disconnect(self):
        """Disconnect cleanly."""
        self.ib.disconnect()
        logger.info("Disconnected from IBKR")

    def is_connected(self) -> bool:
        return self.ib.isConnected()

    # ------------------------------------------------------------------
    # 1. Equity prices — for realized vol computation
    # ------------------------------------------------------------------

    def fetch_historical_prices(
        self, symbols: list[str] | None = None, days: int = 252
    ) -> pd.DataFrame:
        """
        Fetch daily close prices for all constituents + SPX.
        Returns a DataFrame indexed by date with one column per symbol.
        """
        if symbols is None:
            symbols = ["SPX_INDEX"] + constituent_symbols()

        frames = {}
        for sym in symbols:
            contract = UNIVERSE[sym]["contract"]
            self.ib.qualifyContracts(contract)

            duration = f"{days} D" if days <= 365 else f"{math.ceil(days / 365)} Y"
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting="1 day",
                whatToShow="TRADES" if sym != "SPX_INDEX" else "TRADES",
                useRTH=True,
                formatDate=1,
            )
            if bars:
                df = util.df(bars)[["date", "close"]].rename(columns={"close": sym})
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                frames[sym] = df[sym]
                logger.debug("Fetched %d bars for %s", len(bars), sym)
            else:
                logger.warning("No historical data for %s", sym)

            self.ib.sleep(0.5)  # respect rate limits

        return pd.DataFrame(frames).sort_index()

    # ------------------------------------------------------------------
    # 2. Options chains — for implied vol surface
    # ------------------------------------------------------------------

    def fetch_option_chain(self, symbol: str) -> pd.DataFrame:
        """
        Fetch the full option chain for a symbol.
        Returns DataFrame with columns matching option_ticks schema.
        """
        contract = UNIVERSE[symbol]["contract"]
        self.ib.qualifyContracts(contract)

        chains = self.ib.reqSecDefOptParams(
            contract.symbol,
            "",
            contract.secType,
            contract.conId,
        )
        if not chains:
            logger.warning("No option chains for %s", symbol)
            return pd.DataFrame()

        # Use SMART exchange chain
        chain = next((c for c in chains if c.exchange == "SMART"), chains[0])

        rows = []
        for tenor_months in TENOR_MONTHS:
            expiry = _nearest_expiry(sorted(chain.expirations), tenor_months)
            if not expiry:
                continue

            # Get strikes near ATM
            ticker = self.ib.reqMktData(contract, "", False, False)
            self.ib.sleep(2)
            spot = ticker.marketPrice()
            if not spot or math.isnan(spot):
                spot = ticker.close
            self.ib.cancelMktData(contract)

            if not spot or math.isnan(spot):
                logger.warning("No spot price for %s, skipping chain", symbol)
                continue

            # Filter strikes to +/- 30% of spot
            strikes = sorted(
                s for s in chain.strikes
                if spot * 0.70 <= s <= spot * 1.30
            )

            for strike in strikes:
                for right in ["P", "C"]:
                    if symbol == "SPX_INDEX":
                        opt = spx_option(expiry, strike, right)
                    else:
                        opt = equity_option(symbol, expiry, strike, right)

                    try:
                        self.ib.qualifyContracts(opt)
                    except Exception:
                        continue

                    tick = self.ib.reqMktData(opt, "106", False, False)  # 106 = IV
                    self.ib.sleep(0.3)

                    iv = tick.modelGreeks.impliedVol if tick.modelGreeks else None
                    delta = tick.modelGreeks.delta if tick.modelGreeks else None
                    gamma = tick.modelGreeks.gamma if tick.modelGreeks else None
                    vega = tick.modelGreeks.vega if tick.modelGreeks else None
                    theta = tick.modelGreeks.theta if tick.modelGreeks else None

                    rows.append({
                        "ts": datetime.now(timezone.utc),
                        "underlying": symbol,
                        "expiry": expiry,
                        "strike": strike,
                        "right": right,
                        "bid": tick.bid if tick.bid != -1 else None,
                        "ask": tick.ask if tick.ask != -1 else None,
                        "last": tick.last if tick.last != -1 else None,
                        "volume": tick.volume if tick.volume != -1 else None,
                        "open_interest": None,  # requires separate request
                        "implied_vol": iv,
                        "delta": delta,
                        "gamma": gamma,
                        "vega": vega,
                        "theta": theta,
                        "spot": spot,
                        "tenor_months": tenor_months,
                    })

                    self.ib.cancelMktData(opt)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 3. VIX data — for regime signals
    # ------------------------------------------------------------------

    def fetch_vix_data(self) -> dict:
        """
        Fetch current VIX spot, VIX futures term structure, and VVIX.
        Returns a dict with keys matching vol_regime_signals columns.
        """
        # VIX spot
        vix_contract = UNIVERSE["SPX_INDEX"]["contract"]  # We'll use the VIX index
        vix_idx = Index("VIX", "CBOE")
        self.ib.qualifyContracts(vix_idx)
        vix_tick = self.ib.reqMktData(vix_idx, "", False, False)
        self.ib.sleep(2)
        vix_spot = vix_tick.marketPrice()
        self.ib.cancelMktData(vix_idx)

        # VVIX
        vvix_idx = Index("VVIX", "CBOE")
        try:
            self.ib.qualifyContracts(vvix_idx)
            vvix_tick = self.ib.reqMktData(vvix_idx, "", False, False)
            self.ib.sleep(2)
            vvix = vvix_tick.marketPrice()
            self.ib.cancelMktData(vvix_idx)
        except Exception:
            vvix = None
            logger.warning("Could not fetch VVIX")

        # VIX futures — front and second month
        from datetime import date

        today = date.today()
        # VIX futures expire on 3rd Wednesday, approximate with monthly expiries
        front_exp = (today + timedelta(days=20)).strftime("%Y%m") + "01"
        second_exp = (today + timedelta(days=50)).strftime("%Y%m") + "01"

        vix_front = vix_spot  # fallback
        vix_second = vix_spot

        for i, exp_approx in enumerate([front_exp, second_exp]):
            try:
                fut = vix_future(exp_approx)
                self.ib.qualifyContracts(fut)
                fut_tick = self.ib.reqMktData(fut, "", False, False)
                self.ib.sleep(2)
                price = fut_tick.marketPrice()
                self.ib.cancelMktData(fut)
                if price and not math.isnan(price):
                    if i == 0:
                        vix_front = price
                    else:
                        vix_second = price
            except Exception:
                logger.warning("Could not fetch VIX future %s", exp_approx)

        term_slope = ((vix_second - vix_front) / vix_front * 100) if vix_front else 0.0

        return {
            "vix_front_month": vix_front if vix_front and not math.isnan(vix_front) else None,
            "vix_second_month": vix_second if vix_second and not math.isnan(vix_second) else None,
            "term_structure_slope": term_slope,
            "vvix": vvix if vvix and not math.isnan(vvix) else None,
        }

    # ------------------------------------------------------------------
    # 4. Streaming — real-time tick subscription
    # ------------------------------------------------------------------

    def subscribe_equity_ticks(self, symbols: list[str], callback):
        """
        Subscribe to real-time ticks for a list of equity symbols.
        callback(symbol, tick_data) is called on each update.
        """
        for sym in symbols:
            contract = UNIVERSE[sym]["contract"]
            self.ib.qualifyContracts(contract)
            ticker = self.ib.reqMktData(contract, "", False, False)
            ticker.updateEvent += lambda t, s=sym: callback(s, t)
            logger.info("Subscribed to real-time ticks for %s", sym)

    def unsubscribe_all(self):
        """Cancel all market data subscriptions."""
        for ticker in self.ib.tickers():
            self.ib.cancelMktData(ticker.contract)
        logger.info("Unsubscribed from all market data")
