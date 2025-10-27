# MT5 ML Trading EA - Quick Start

## 5-Minute Setup (For Experienced Traders)

### 1. Install EA
```
1. Open MT5 → Press F4 (MetaEditor)
2. File → Open → Select MT5_ML_Trading_EA.mq5
3. Press F7 to compile
4. Restart MT5
```

### 2. Attach to Chart
```
1. Open EURUSD chart → Set M5 timeframe
2. Navigator (Ctrl+N) → Expert Advisors → MT5_ML_Trading_EA
3. Drag EA onto chart
4. Enable AutoTrading (F7) - button turns GREEN
```

### 3. Configure Settings
**Minimum Required Settings:**
```
Lot Size: 0.01 (for $1000 account)
Stop Loss Pips: 20
Take Profit Pips: 30
Max Spread Pips: 3
```

### 4. Start Trading
- EA will display info panel on chart
- Watch for LONG/SHORT signals
- Check "Experts" tab for logs

---

## Recommended Settings by Account Size

### $500 Account
```
Lot Size: 0.01
Risk Percent: 1%
Max Positions: 1
Stop Loss: 20 pips
```

### $1,000 Account
```
Lot Size: 0.02
Risk Percent: 2%
Max Positions: 1
Stop Loss: 20 pips
```

### $5,000 Account
```
Lot Size: 0.1
Risk Percent: 2%
Max Positions: 2
Stop Loss: 20 pips
```

### $10,000+ Account
```
Lot Size: 0.2
Risk Percent: 2%
Max Positions: 3
Stop Loss: 20 pips
```

---

## Best Forex Pairs for This EA

### Tier 1 (Recommended)
- **EURUSD**: Most liquid, tight spreads
- **GBPUSD**: Good volatility, clear trends
- **USDJPY**: Low spreads, smooth movements

### Tier 2 (Advanced)
- **AUDUSD**: Good for trending markets
- **NZDUSD**: Similar to AUDUSD
- **USDCAD**: Oil-correlated, good volatility

### Tier 3 (High Risk)
- **XAUUSD** (Gold): Very volatile, adjust SL/TP
- **GBPJPY**: High volatility, wider spreads
- **Exotic pairs**: Not recommended

---

## Optimal Trading Times (GMT)

### Best Sessions
```
London Open:     07:00 - 11:00 GMT  ✓ BEST
London/NY:       12:00 - 16:00 GMT  ✓ BEST
NY Session:      13:00 - 17:00 GMT  ✓ GOOD
Asian Close:     05:00 - 07:00 GMT  → OK
```

### Avoid
```
Asian Session:   00:00 - 05:00 GMT  ✗ Low liquidity
Late NY:         20:00 - 23:00 GMT  ✗ Low volume
Weekends:        Closed             ✗ No trading
```

---

## Key Parameters Cheat Sheet

| Parameter | Conservative | Balanced | Aggressive |
|-----------|-------------|----------|------------|
| Lot Size | 0.01 | 0.05 | 0.1+ |
| Risk % | 1% | 2% | 3-5% |
| Stop Loss | 25 pips | 20 pips | 15 pips |
| Take Profit | 40 pips | 30 pips | 25 pips |
| Long Threshold | 0.70 | 0.60 | 0.55 |
| Short Threshold | 0.70 | 0.60 | 0.55 |
| Max Positions | 1 | 2 | 3 |
| Max Spread | 2 pips | 3 pips | 4 pips |

---

## How the ML Signals Work

### LONG Signal Triggers When:
- ✓ Fast MA crosses above Slow MA
- ✓ RSI bounces from oversold (< 30)
- ✓ Price bounces from lower Bollinger Band
- ✓ 5-bar bullish momentum
- ✓ Higher timeframe confirms uptrend
- **Score > 0.60** = LONG signal

### SHORT Signal Triggers When:
- ✓ Fast MA crosses below Slow MA
- ✓ RSI reverses from overbought (> 70)
- ✓ Price rejects upper Bollinger Band
- ✓ 5-bar bearish momentum
- ✓ Higher timeframe confirms downtrend
- **Score > 0.60** = SHORT signal

---

## Troubleshooting Quick Fixes

### EA Not Trading?
```
1. Check AutoTrading is GREEN (press F7)
2. Check Experts tab for errors
3. Verify current spread < Max Spread
4. Check if within trading hours
5. Ensure AllowLong or AllowShort = true
```

### Too Many Trades?
```
1. Increase Long/Short Threshold to 0.65
2. Reduce Max Positions to 1
3. Enable stricter time filters
4. Increase Min Bars requirement
```

### Trailing Stop Not Working?
```
1. Enable: UseTrailingStop = true
2. Position must be in profit
3. Increase TrailingStopPips (broker minimum)
```

### "Not Enough Money" Error?
```
1. Reduce Lot Size
2. Check account balance
3. Close other positions
4. Verify margin requirements
```

---

## Performance Monitoring

### Daily Checks
- [ ] Check open positions
- [ ] Review Experts tab for errors
- [ ] Verify EA is active (smiley face icon)
- [ ] Monitor account equity

### Weekly Checks
- [ ] Calculate win rate (should be 55%+)
- [ ] Review profit factor (should be 1.3+)
- [ ] Check max drawdown (should be < 20%)
- [ ] Analyze trade log

### Monthly Checks
- [ ] Compare to backtest results
- [ ] Optimize parameters if needed
- [ ] Review broker costs (spreads/commissions)
- [ ] Plan next month's strategy

---

## Red Flags - Stop Trading If:

🚨 **Stop Immediately If:**
- Win rate drops below 40%
- Drawdown exceeds 30%
- Profit factor < 1.0
- 5+ consecutive losses
- Unusual broker behavior

⚠️ **Review Strategy If:**
- Win rate 40-50%
- Drawdown 20-30%
- Profit factor 1.0-1.2
- Results differ from backtest
- Spread consistently high

---

## Python ML Server (Advanced)

### Start Server
```bash
cd AlgoT
python MT5_ML_Server.py
```

### Verify Server Running
```
✓ ML Server listening on 127.0.0.1:9090
Waiting for MT5 connections...
```

### Benefits
- Full ML ensemble (6 models)
- Better accuracy
- Real-time predictions
- Continuous learning

### When to Use
- After mastering basic EA
- Want maximum performance
- Have Python experience
- Trading larger accounts

---

## Example Trade Log (What Success Looks Like)

```
Day 1:  +2.3% | 3 wins, 1 loss | Win rate: 75%
Day 2:  +1.8% | 2 wins, 1 loss | Win rate: 67%
Day 3:  -0.5% | 1 win, 2 loss | Win rate: 33%
Day 4:  +2.1% | 3 wins, 0 loss | Win rate: 100%
Day 5:  +1.2% | 2 wins, 1 loss | Win rate: 67%
────────────────────────────────────────────
Week:   +6.9% | 11 wins, 5 losses | 69% ✓
```

**Good Performance Indicators:**
- Positive weekly return
- Win rate > 55%
- More winning days than losing
- Controlled drawdowns
- Consistent with backtest

---

## Safety Rules (READ BEFORE LIVE TRADING)

### The 3 Golden Rules
1. **Always use stop loss** - Never trade without protection
2. **Risk max 2% per trade** - Preserve capital
3. **Test on demo first** - Minimum 2 weeks

### Before Going Live
- [ ] 2+ weeks profitable on demo
- [ ] Win rate above 55%
- [ ] Max drawdown under 20%
- [ ] Understand all parameters
- [ ] Have monitoring plan
- [ ] Start with minimum lot size
- [ ] Accept you may lose money

### Emergency Procedure
If things go wrong:
1. Press F7 to disable AutoTrading
2. Close all positions manually
3. Review logs and settings
4. Return to demo account
5. Identify the issue
6. Test fix on demo
7. Only then resume live trading

---

## Resources

📚 **Full Guide**: See `MT5_SETUP_GUIDE.md`
🐍 **Python Bridge**: See `MT5_ML_Server.py`
💻 **Source Code**: `MT5_ML_Trading_EA.mq5`
📊 **Python Algo**: `ML-Enhanced-algo.py`

---

## Support

- **Bugs**: Open GitHub issue
- **Questions**: Check MT5_SETUP_GUIDE.md
- **Updates**: Watch GitHub repo

---

**Happy Trading! Remember: Test first, trade small, think long-term!** 🚀

---

*Version 1.0 | Last Updated: October 2025*
