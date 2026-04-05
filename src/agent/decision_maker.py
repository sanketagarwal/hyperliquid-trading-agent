"""Decision-making agent that orchestrates LLM prompts and indicator lookups.

Uses the Anthropic Claude API directly for trade decisions.
"""

import asyncio
import anthropic
from src.config_loader import CONFIG
from src.indicators.local_indicators import compute_all, last_n, latest
import json
import logging
from datetime import datetime


class TradingAgent:
    """High-level trading agent that delegates reasoning to Claude."""

    def __init__(self, hyperliquid=None):
        self.model = CONFIG["llm_model"]
        self.client = anthropic.Anthropic(api_key=CONFIG["anthropic_api_key"])
        self.hyperliquid = hyperliquid
        self.sanitize_model = CONFIG.get("sanitize_model") or "claude-haiku-4-5-20251001"
        self.max_tokens = int(CONFIG.get("max_tokens") or 4096)

    def decide_trade(self, assets, context):
        """Decide for multiple assets in one call."""
        return self._decide(context, assets=assets)

    def _decide(self, context, assets):
        """Dispatch decision request to Claude and enforce output contract."""
        system_prompt = (
            "You are a rigorous QUANTITATIVE TRADER and interdisciplinary MATHEMATICIAN-ENGINEER optimizing risk-adjusted returns for perpetual futures under real execution, margin, and funding constraints.\n"
            "You will receive market + account context for SEVERAL assets, including:\n"
            f"- assets = {json.dumps(list(assets))}\n"
            "- per-asset intraday (5m) and higher-timeframe (4h) metrics\n"
            "- Active Trades with Exit Plans\n"
            "- Recent Trading History\n"
            "- Risk management limits (hard-enforced by the system, not just guidelines)\n\n"
            "Always use the 'current time' provided in the user message to evaluate any time-based conditions, such as cooldown expirations or timed exit plans.\n\n"
            "Your goal: make decisive, first-principles decisions per asset that minimize churn while capturing edge.\n\n"
            "Aggressively pursue setups where calculated risk is outweighed by expected edge; size positions so downside is controlled while upside remains meaningful.\n\n"
            "Core policy (low-churn, position-aware)\n"
            "1) Respect prior plans: If an active trade has an exit_plan with explicit invalidation (e.g., \"close if 4h close above EMA50\"), DO NOT close or flip early unless that invalidation (or a stronger one) has occurred.\n"
            "2) Hysteresis: Require stronger evidence to CHANGE a decision than to keep it. Only flip direction if BOTH:\n"
            "   a) Higher-timeframe structure supports the new direction (e.g., 4h EMA20 vs EMA50 and/or MACD regime), AND\n"
            "   b) Intraday structure confirms with a decisive break beyond ~0.5×ATR (recent) and momentum alignment (MACD or RSI slope).\n"
            "   Otherwise, prefer HOLD or adjust TP/SL.\n"
            "3) Cooldown: After opening, adding, reducing, or flipping, impose a self-cooldown of at least 3 bars of the decision timeframe (e.g., 3×5m = 15m) before another direction change, unless a hard invalidation occurs. Encode this in exit_plan (e.g., \"cooldown_bars:3 until 2025-10-19T15:55Z\"). You must honor your own cooldowns on future cycles.\n"
            "4) Funding is a tilt, not a trigger: Do NOT open/close/flip solely due to funding unless expected funding over your intended holding horizon meaningfully exceeds expected edge (e.g., > ~0.25×ATR). Consider that funding accrues discretely and slowly relative to 5m bars.\n"
            "5) Overbought/oversold ≠ reversal by itself: Treat RSI extremes as risk-of-pullback. You need structure + momentum confirmation to bet against trend. Prefer tightening stops or taking partial profits over instant flips.\n"
            "6) Prefer adjustments over exits: If the thesis weakens but is not invalidated, first consider: tighten stop (e.g., to a recent swing or ATR multiple), trail TP, or reduce size. Flip only on hard invalidation + fresh confluence.\n\n"
            "Decision discipline (per asset)\n"
            "- Choose one: buy / sell / hold.\n"
            "- Proactively harvest profits when price action presents a clear, high-quality opportunity that aligns with your thesis.\n"
            "- You control allocation_usd (but the system will cap it — see risk limits below).\n"
            "- Order type: set order_type to \"market\" for immediate execution, or \"limit\" for resting orders.\n"
            "  • For limit orders, you MUST set limit_price. Use limit orders when you want better entry prices (e.g., buying a dip, selling a bounce).\n"
            "  • For market orders, limit_price should be null.\n"
            "  • Default is \"market\" if omitted.\n"
            "- TP/SL sanity:\n"
            "  • BUY: tp_price > current_price, sl_price < current_price\n"
            "  • SELL: tp_price < current_price, sl_price > current_price\n"
            "  If sensible TP/SL cannot be set, use null and explain the logic. A mandatory SL will be auto-applied if you don't set one.\n"
            "- exit_plan must include at least ONE explicit invalidation trigger and may include cooldown guidance you will follow later.\n\n"
            "Leverage policy (perpetual futures)\n"
            "- You can use leverage, but the system enforces a hard cap. Stay within the limits.\n"
            "- In high volatility (elevated ATR) or during funding spikes, reduce or avoid leverage.\n"
            "- Treat allocation_usd as notional exposure; keep it consistent with safe leverage and available margin.\n\n"
            "Tool usage\n"
            "- Use the fetch_indicator tool whenever an additional datapoint could sharpen your thesis; parameters: indicator (ema/sma/rsi/macd/bbands/atr/adx/obv/vwap/stoch_rsi/all), asset (e.g. \"BTC\", \"OIL\", \"GOLD\"), interval (\"5m\"/\"4h\"), optional period.\n"
            "- Indicators are computed locally from Hyperliquid candle data — works for ALL perp markets (crypto, commodities, indices).\n"
            "- Incorporate tool findings into your reasoning, but NEVER paste raw tool responses into the final JSON — summarize the insight instead.\n"
            "- Use tools to upgrade your analysis; lack of confidence is a cue to query them before deciding.\n\n"
            "Reasoning recipe (first principles)\n"
            "- Structure (trend, EMAs slope/cross, HH/HL vs LH/LL), Momentum (MACD regime, RSI slope), Liquidity/volatility (ATR, volume), Positioning tilt (funding, OI).\n"
            "- Favor alignment across 4h and 5m. Counter-trend scalps require stronger intraday confirmation and tighter risk.\n\n"
            "Output contract\n"
            "- Output ONLY a strict JSON object (no markdown, no code fences) with exactly two properties:\n"
            "  • \"reasoning\": long-form string capturing detailed, step-by-step analysis.\n"
            "  • \"trade_decisions\": array ordered to match the provided assets list.\n"
            "- Each item inside trade_decisions must contain the keys: asset, action, allocation_usd, order_type, limit_price, tp_price, sl_price, exit_plan, rationale.\n"
            "  • order_type: \"market\" (default) or \"limit\"\n"
            "  • limit_price: required if order_type is \"limit\", null otherwise\n"
            "- Do not emit Markdown or any extra properties.\n"
        )

        tools = [{
            "name": "fetch_indicator",
            "description": (
                "Fetch technical indicators computed locally from Hyperliquid candle data. "
                "Works for ALL Hyperliquid perp markets including crypto (BTC, ETH, SOL), "
                "commodities (OIL, GOLD, SILVER), indices (SPX), and more. "
                "Available indicators: ema, sma, rsi, macd, bbands, atr, adx, obv, vwap, stoch_rsi, all. "
                "Returns the latest values and recent series."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "indicator": {
                        "type": "string",
                        "enum": ["ema", "sma", "rsi", "macd", "bbands", "atr", "adx", "obv", "vwap", "stoch_rsi", "all"],
                    },
                    "asset": {
                        "type": "string",
                        "description": "Hyperliquid asset symbol, e.g. BTC, ETH, OIL, GOLD, SPX",
                    },
                    "interval": {
                        "type": "string",
                        "enum": ["1m", "5m", "15m", "1h", "4h", "1d"],
                    },
                    "period": {
                        "type": "integer",
                        "description": "Indicator period (default varies by indicator)",
                    },
                },
                "required": ["indicator", "asset", "interval"],
            },
        }]

        messages = [{"role": "user", "content": context}]

        def _log_request(model, messages_to_log):
            with open("llm_requests.log", "a", encoding="utf-8") as f:
                f.write(f"\n\n=== {datetime.now()} ===\n")
                f.write(f"Model: {model}\n")
                f.write(f"Messages count: {len(messages_to_log)}\n")
                # Log last message content (truncated)
                last = messages_to_log[-1]
                content_str = str(last.get("content", ""))[:500]
                f.write(f"Last message role: {last.get('role')}\n")
                f.write(f"Last message content (truncated): {content_str}\n")

        def _call_claude(msgs, use_tools=True):
            """Make a Claude API call with optional tool use."""
            _log_request(self.model, msgs)
            kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": system_prompt,
                "messages": msgs,
            }
            if use_tools:
                kwargs["tools"] = tools
            if CONFIG.get("thinking_enabled"):
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": int(CONFIG.get("thinking_budget_tokens") or 10000),
                }
                # When thinking is enabled, max_tokens must be larger
                kwargs["max_tokens"] = max(self.max_tokens, 16000)

            response = self.client.messages.create(**kwargs)
            logging.info("Claude response: stop_reason=%s, usage=%s",
                        response.stop_reason, response.usage)
            with open("llm_requests.log", "a", encoding="utf-8") as f:
                f.write(f"Response stop_reason: {response.stop_reason}\n")
                f.write(f"Usage: input={response.usage.input_tokens}, output={response.usage.output_tokens}\n")
            return response

        def _handle_tool_call(tool_name, tool_input):
            """Execute a tool call and return the result string."""
            if tool_name != "fetch_indicator":
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

            try:
                asset = tool_input["asset"]
                interval = tool_input["interval"]
                indicator = tool_input["indicator"]

                # Fetch candles from Hyperliquid
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        candles = pool.submit(
                            asyncio.run,
                            self.hyperliquid.get_candles(asset, interval, 100)
                        ).result(timeout=30)
                else:
                    candles = asyncio.run(self.hyperliquid.get_candles(asset, interval, 100))

                all_indicators = compute_all(candles)

                if indicator == "all":
                    result = {k: {"latest": latest(v) if isinstance(v, list) else v,
                                  "series": last_n(v, 10) if isinstance(v, list) else v}
                              for k, v in all_indicators.items()}
                elif indicator == "macd":
                    result = {
                        "macd": {"latest": latest(all_indicators.get("macd", [])), "series": last_n(all_indicators.get("macd", []), 10)},
                        "signal": {"latest": latest(all_indicators.get("macd_signal", [])), "series": last_n(all_indicators.get("macd_signal", []), 10)},
                        "histogram": {"latest": latest(all_indicators.get("macd_histogram", [])), "series": last_n(all_indicators.get("macd_histogram", []), 10)},
                    }
                elif indicator == "bbands":
                    result = {
                        "upper": {"latest": latest(all_indicators.get("bbands_upper", [])), "series": last_n(all_indicators.get("bbands_upper", []), 10)},
                        "middle": {"latest": latest(all_indicators.get("bbands_middle", [])), "series": last_n(all_indicators.get("bbands_middle", []), 10)},
                        "lower": {"latest": latest(all_indicators.get("bbands_lower", [])), "series": last_n(all_indicators.get("bbands_lower", []), 10)},
                    }
                elif indicator in ("ema", "sma"):
                    period = tool_input.get("period", 20)
                    from src.indicators.local_indicators import ema as _ema, sma as _sma
                    closes = [c["close"] for c in candles]
                    series = _ema(closes, period) if indicator == "ema" else _sma(closes, period)
                    result = {"latest": latest(series), "series": last_n(series, 10), "period": period}
                elif indicator == "rsi":
                    period = tool_input.get("period", 14)
                    from src.indicators.local_indicators import rsi as _rsi
                    series = _rsi(candles, period)
                    result = {"latest": latest(series), "series": last_n(series, 10), "period": period}
                elif indicator == "atr":
                    period = tool_input.get("period", 14)
                    from src.indicators.local_indicators import atr as _atr
                    series = _atr(candles, period)
                    result = {"latest": latest(series), "series": last_n(series, 10), "period": period}
                else:
                    key_map = {"adx": "adx", "obv": "obv", "vwap": "vwap", "stoch_rsi": "stoch_rsi"}
                    mapped = key_map.get(indicator, indicator)
                    series = all_indicators.get(mapped, [])
                    result = {"latest": latest(series) if isinstance(series, list) else series,
                              "series": last_n(series, 10) if isinstance(series, list) else series}

                return json.dumps(result, default=str)
            except Exception as ex:
                logging.error("Tool call error: %s", ex)
                return json.dumps({"error": str(ex)})

        def _sanitize_output(raw_content: str, assets_list):
            """Use a cheap Claude model to normalize malformed output."""
            try:
                response = self.client.messages.create(
                    model=self.sanitize_model,
                    max_tokens=2048,
                    system=(
                        "You are a strict JSON normalizer. Return ONLY a JSON object with two keys: "
                        "\"reasoning\" (string) and \"trade_decisions\" (array). "
                        "Each trade_decisions item must have: asset, action (buy/sell/hold), "
                        "allocation_usd (number), order_type (\"market\" or \"limit\"), "
                        "limit_price (number or null), tp_price (number or null), sl_price (number or null), "
                        "exit_plan (string), rationale (string). "
                        f"Valid assets: {json.dumps(list(assets_list))}. "
                        "If input is wrapped in markdown or has prose, extract just the JSON. Do not add fields."
                    ),
                    messages=[{"role": "user", "content": raw_content}],
                )
                content = ""
                for block in response.content:
                    if block.type == "text":
                        content += block.text
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "trade_decisions" in parsed:
                    return parsed
                return {"reasoning": "", "trade_decisions": []}
            except Exception as se:
                logging.error("Sanitize failed: %s", se)
                return {"reasoning": "", "trade_decisions": []}

        # Main loop: up to 6 iterations to handle tool calls
        for iteration in range(6):
            try:
                response = _call_claude(messages)
            except anthropic.APIError as e:
                logging.error("Claude API error: %s", e)
                with open("llm_requests.log", "a", encoding="utf-8") as f:
                    f.write(f"API Error: {e}\n")
                break

            # Check if the response contains tool use
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            if tool_use_blocks and response.stop_reason == "tool_use":
                # Build assistant message with all content blocks
                assistant_content = []
                for block in response.content:
                    if block.type == "text":
                        assistant_content.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })
                    elif block.type == "thinking":
                        assistant_content.append({
                            "type": "thinking",
                            "thinking": block.thinking,
                        })
                messages.append({"role": "assistant", "content": assistant_content})

                # Process each tool call
                tool_results = []
                for block in tool_use_blocks:
                    result_str = _handle_tool_call(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })
                messages.append({"role": "user", "content": tool_results})
                continue

            # No tool calls — parse the text response as JSON
            raw_text = ""
            for block in text_blocks:
                raw_text += block.text

            if not raw_text.strip():
                logging.error("Empty response from Claude")
                break

            try:
                parsed = json.loads(raw_text)
                if not isinstance(parsed, dict):
                    logging.error("Expected dict, got: %s; attempting sanitize", type(parsed))
                    return _sanitize_output(raw_text, assets)

                reasoning_text = parsed.get("reasoning", "") or ""
                decisions = parsed.get("trade_decisions")

                if isinstance(decisions, list):
                    normalized = []
                    for item in decisions:
                        if isinstance(item, dict):
                            item.setdefault("allocation_usd", 0.0)
                            item.setdefault("order_type", "market")
                            item.setdefault("limit_price", None)
                            item.setdefault("tp_price", None)
                            item.setdefault("sl_price", None)
                            item.setdefault("exit_plan", "")
                            item.setdefault("rationale", "")
                            normalized.append(item)
                    return {"reasoning": reasoning_text, "trade_decisions": normalized}

                logging.error("trade_decisions missing or invalid; attempting sanitize")
                sanitized = _sanitize_output(raw_text, assets)
                if sanitized.get("trade_decisions"):
                    return sanitized
                return {"reasoning": reasoning_text, "trade_decisions": []}

            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logging.error("JSON parse error: %s, content: %s", e, raw_text[:200])
                sanitized = _sanitize_output(raw_text, assets)
                if sanitized.get("trade_decisions"):
                    return sanitized
                return {
                    "reasoning": "Parse error",
                    "trade_decisions": [{
                        "asset": a,
                        "action": "hold",
                        "allocation_usd": 0.0,
                        "tp_price": None,
                        "sl_price": None,
                        "exit_plan": "",
                        "rationale": "Parse error"
                    } for a in assets]
                }

        # Exhausted tool loop
        return {
            "reasoning": "tool loop cap",
            "trade_decisions": [{
                "asset": a,
                "action": "hold",
                "allocation_usd": 0.0,
                "tp_price": None,
                "sl_price": None,
                "exit_plan": "",
                "rationale": "tool loop cap"
            } for a in assets]
        }
