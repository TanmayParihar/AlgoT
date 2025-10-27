# MetaTrader 5 ML Trading EA - Complete Setup Guide

## Overview

This guide will help you install and configure the ML-Enhanced Trading Expert Advisor (EA) for MetaTrader 5. The system supports automated day trading on forex pairs like EURUSD, GBPUSD, etc., with advanced ML-based signal generation.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation Steps](#installation-steps)
3. [Configuration](#configuration)
4. [Usage Modes](#usage-modes)
5. [Parameters Explained](#parameters-explained)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

---

## Requirements

### MetaTrader 5
- **Platform**: MetaTrader 5 (build 3000 or higher)
- **Account Type**: Any (Demo recommended for testing)
- **Broker**: Any MT5 broker supporting forex trading
- **Timeframe**: M5 (5-minute) recommended for day trading

### Python Server (Optional - for Advanced ML)
- **Python**: 3.8 or higher
- **Libraries**:
  ```bash
  pip install numpy pandas scikit-learn
  ```

### System Requirements
- **OS**: Windows 10/11 (for MT5)
- **RAM**: 4GB minimum, 8GB recommended
- **Internet**: Stable connection required

---

## Installation Steps

### Step 1: Install the Expert Advisor

1. **Open MetaTrader 5**
   - Launch your MT5 platform

2. **Open MetaEditor**
   - Press `F4` or click Tools → MetaQuotes Language Editor

3. **Create New Expert Advisor**
   - File → New → Expert Advisor (template)
   - Or copy the provided `MT5_ML_Trading_EA.mq5` file

4. **Copy the Code**
   - Open `MT5_ML_Trading_EA.mq5` from this repository
   - Copy the entire code
   - Paste into MetaEditor

5. **Compile the EA**
   - Click "Compile" button (F7)
   - Check for errors in the "Errors" tab
   - Should see: "0 error(s), 0 warning(s)"

6. **Locate the EA**
   - The EA will be saved in: `MQL5/Experts/MT5_ML_Trading_EA.ex5`

### Step 2: Attach EA to Chart

1. **Open a Chart**
   - File → New Chart → Select your pair (e.g., EURUSD)
   - Set timeframe to M5 (5-minute)

2. **Attach the EA**
   - In Navigator window (Ctrl+N), expand "Expert Advisors"
   - Find "MT5_ML_Trading_EA"
   - Drag and drop onto the chart

3. **Enable Auto-Trading**
   - Click "AutoTrading" button in toolbar (or F7)
   - Button should turn GREEN

4. **Allow DLL Imports** (if using Python bridge)
   - In EA settings, go to "Common" tab
   - Check "Allow DLL imports"

### Step 3: Python ML Server Setup (Optional)

This is for **advanced users** who want full ML predictions. The EA works standalone without Python.

1. **Install Python Dependencies**
   ```bash
   cd AlgoT
   pip install numpy pandas scikit-learn
   ```

2. **Train or Load Models**
   ```bash
   python MT5_ML_Server.py
   ```

   - First run will generate sample models
   - For real data, replace sample data with your historical data

3. **Start the Server**
   ```bash
   python MT5_ML_Server.py
   ```

   - Server starts on `127.0.0.1:9090`
   - Keep this running while trading

4. **Configure EA for Python Mode**
   - This requires custom DLL implementation (advanced)
   - By default, EA uses built-in ML logic

---

## Configuration

### Recommended Settings for EURUSD/GBPUSD

```
=== Trading Settings ===
Lot Size: 0.1 (or 0.01 for micro accounts)
Risk Percent: 2.0%
Max Positions: 1
Allow Long: true
Allow Short: true

=== Risk Management ===
Stop Loss Pips: 20
Take Profit Pips: 30
Use Trailing Stop: true
Trailing Stop Pips: 15
Max Spread Pips: 3.0

=== ML Model Settings ===
Fast MA: 12
Slow MA: 26
RSI Period: 14
RSI Overbought: 70
RSI Oversold: 30
BB Period: 20
BB Deviation: 2.0

=== ML Signal Settings ===
Long Threshold: 0.60
Short Threshold: 0.60
Min Bars For Signal: 50

=== Time Filters ===
Use Time Filter: true
Start Hour: 7 (for London session)
End Hour: 16 (before NY close)
Trade Monday-Friday: true
```

### Settings for Different Pairs

**GBPUSD (More Volatile)**
- Stop Loss: 25 pips
- Take Profit: 40 pips
- Max Spread: 4 pips
- Trailing Stop: 20 pips

**USDJPY (Less Volatile)**
- Stop Loss: 15 pips
- Take Profit: 25 pips
- Max Spread: 2 pips
- Trailing Stop: 12 pips

**Gold (XAUUSD) - Very Volatile**
- Stop Loss: 50 pips
- Take Profit: 80 pips
- Max Spread: 10 pips
- Trailing Stop: 40 pips

---

## Usage Modes

### Mode 1: Standalone (Recommended for Beginners)

The EA works completely standalone using built-in technical indicators:
- MA crossovers
- RSI divergence
- Bollinger Bands
- ATR volatility
- Multi-timeframe confirmation

**Pros:**
- Easy setup
- No external dependencies
- Fast execution
- Reliable

**Cons:**
- Simpler ML logic
- No deep learning

### Mode 2: Python Bridge (Advanced)

Connect EA to Python ML server for full predictions:
- 6 ML models (GB, RF, NN x 2)
- Advanced feature engineering
- Sentiment analysis integration

**Pros:**
- Full ML power
- Better accuracy
- Continuous model updates

**Cons:**
- Requires Python setup
- More complex
- Additional latency

---

## Parameters Explained

### Trading Settings

**Lot Size**
- Fixed lot size per trade
- Example: 0.1 = 10,000 units (mini lot)
- Use 0.01 for micro accounts

**Risk Percent**
- Percentage of account to risk per trade
- 2% is standard
- Conservative: 1%, Aggressive: 3-5%

**Max Positions**
- Maximum concurrent trades
- 1 = safer, focus on quality
- 2-3 = more opportunities, higher risk

### Risk Management

**Stop Loss Pips**
- Distance from entry to stop loss
- Protects against large losses
- Adjust based on pair volatility

**Take Profit Pips**
- Distance from entry to take profit
- Target profit level
- Typically 1.5x stop loss

**Trailing Stop**
- Automatically moves SL in profit direction
- Locks in gains
- Follows price at specified distance

**Max Spread Pips**
- Maximum spread allowed to trade
- Prevents trading during high costs
- Typical: 2-4 pips for major pairs

### ML Signal Settings

**Long/Short Threshold**
- Minimum ML score (0-1) to open position
- 0.60 = 60% confidence required
- Higher = fewer but better trades
- Lower = more trades, lower quality

**Min Bars For Signal**
- Minimum historical bars needed
- Ensures indicators are properly calculated
- 50 is safe minimum

### Time Filters

**Trading Hours**
- Start/End hours in broker time
- Avoid low liquidity periods
- Recommended: 7-16 GMT for forex

**Trading Days**
- Enable/disable specific weekdays
- Avoid Mondays (gap risk)
- Avoid Fridays after 15:00 (weekend risk)

---

## Troubleshooting

### EA Not Trading

**Check 1: AutoTrading Enabled**
- Green "AutoTrading" button must be active
- Press F7 to toggle

**Check 2: EA Inputs**
- Verify AllowLong or AllowShort is true
- Check time filter settings
- Confirm Min Bars requirement met

**Check 3: Spread Too High**
- Current spread > MaxSpreadPips
- Wait for tighter spread or increase limit

**Check 4: Errors Tab**
- Check "Experts" tab for error messages
- Look for "Invalid SL/TP" or "Not enough money"

### Positions Not Opening

**Issue: "Not enough money"**
- Solution: Reduce lot size or increase account balance

**Issue: "Invalid stops"**
- Solution: Increase SL/TP distance (broker minimum)
- Check broker's stop level requirement

**Issue: "Market closed"**
- Solution: Wait for market to open
- Check trading session hours

### EA Opening Too Many Trades

**Solution 1: Increase Thresholds**
- Set Long/Short Threshold to 0.65 or 0.70
- Reduces signal frequency

**Solution 2: Reduce Max Positions**
- Set to 1 for more conservative approach

**Solution 3: Tighten Time Filter**
- Trade only during main sessions
- Avoid volatile news times

### Trailing Stop Not Working

**Check 1: Setting Enabled**
- UseTrailingStop must be true

**Check 2: Position in Profit**
- Trailing only activates when profit > trailing distance

**Check 3: Distance Too Small**
- Broker may have minimum distance requirement
- Increase TrailingStopPips

---

## Best Practices

### Risk Management

1. **Start Small**
   - Begin with 0.01 lots on demo account
   - Test for at least 2 weeks
   - Gradually increase after consistent results

2. **Never Risk More Than 2%**
   - Per trade risk should be 1-2% of account
   - Maximum account risk: 6% (all positions)

3. **Use Stop Losses Always**
   - Never disable stop loss
   - Protects from unexpected events
   - Prevents account blowup

4. **Monitor Drawdown**
   - If equity drops 15-20%, reduce lot size
   - If drops 30%, stop trading and review

### Optimization

1. **Backtest First**
   - Use MT5 Strategy Tester
   - Test on at least 6 months of data
   - Focus on M5 timeframe

2. **Forward Test**
   - Run on demo for 1-2 weeks
   - Verify live conditions match backtest
   - Check slippage and spread

3. **Optimize Parameters**
   - Don't over-optimize (curve fitting)
   - Test on different periods
   - Focus on robustness, not max profit

4. **Multi-Pair Testing**
   - Test on multiple pairs
   - Same settings should work reasonably on all
   - Avoid pair-specific optimization

### Monitoring

1. **Check Daily**
   - Review open positions
   - Check for errors in logs
   - Monitor account equity

2. **Weekly Review**
   - Analyze win rate and profit factor
   - Check if performance matches backtest
   - Adjust if needed

3. **News Awareness**
   - Pause EA before major news (NFP, FOMC, etc.)
   - High volatility can trigger many stops
   - Resume after news settles

### Performance Expectations

**Realistic Targets (Monthly)**
- Win Rate: 55-65%
- Profit Factor: 1.3-1.8
- Monthly Return: 5-15%
- Max Drawdown: 10-20%

**Warning Signs**
- Win rate < 40%: Review strategy
- Profit factor < 1.0: Stop trading
- Drawdown > 30%: Reduce risk or stop
- Sharpe < 0.5: Strategy not working

---

## Safety Checklist

Before going live:

- [ ] Tested on demo account for 2+ weeks
- [ ] Backtest shows consistent profits
- [ ] Stop loss and take profit configured
- [ ] Risk per trade ≤ 2%
- [ ] Max spread filter enabled
- [ ] Time filter configured
- [ ] AutoTrading enabled
- [ ] Sufficient account balance
- [ ] Broker allows automated trading
- [ ] Monitoring plan in place

---

## Support & Resources

### Logs Location
- **Windows**: `AppData\Roaming\MetaQuotes\Terminal\[instance]\MQL5\Logs`
- **Journal**: Check for connection issues
- **Experts**: Check for trading errors

### Useful MT5 Functions
- `Ctrl+N`: Navigator window
- `Ctrl+T`: Terminal window
- `Ctrl+H`: Trade history
- `F4`: MetaEditor
- `F7`: Toggle AutoTrading

### Community
- GitHub Issues: Report bugs or request features
- MT5 Forum: General MT5 questions
- Broker Support: Platform-specific issues

---

## Disclaimer

⚠️ **IMPORTANT DISCLAIMER**

- **Trading involves substantial risk**: You can lose all your invested capital
- **Past performance ≠ future results**: Backtests don't guarantee live profits
- **Demo ≠ Live**: Live conditions include slippage, latency, and psychological factors
- **Not financial advice**: This EA is for educational purposes
- **Test thoroughly**: Always start with demo account
- **Understand the code**: Don't trade what you don't understand

**Use at your own risk. The developers are not responsible for any trading losses.**

---

## License

This EA is open source and provided as-is for educational purposes.

---

**Last Updated**: October 2025
**Version**: 1.0
**Compatibility**: MT5 Build 3000+
