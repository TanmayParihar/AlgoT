//+------------------------------------------------------------------+
//|                                           MT5_ML_Trading_EA.mq5 |
//|                   ML-Enhanced Day Trading Expert Advisor for MT5 |
//|                   Supports LONG/SHORT positions with ML signals  |
//+------------------------------------------------------------------+
#property copyright "ML Trading System"
#property link      "https://github.com/TanmayParihar/AlgoT"
#property version   "1.00"
#property strict

//--- Input parameters
input group "=== Trading Settings ==="
input double   LotSize = 0.1;                  // Lot size per trade
input double   RiskPercent = 2.0;              // Risk per trade (%)
input int      MaxPositions = 1;               // Maximum simultaneous positions
input bool     AllowLong = true;               // Allow LONG positions
input bool     AllowShort = true;              // Allow SHORT positions

input group "=== Risk Management ==="
input double   StopLossPips = 20;              // Stop Loss in pips
input double   TakeProfitPips = 30;            // Take Profit in pips
input bool     UseTrailingStop = true;         // Use trailing stop
input double   TrailingStopPips = 15;          // Trailing stop distance in pips
input double   MaxSpreadPips = 3.0;            // Maximum spread to trade (pips)

input group "=== ML Model Settings ==="
input int      FastMA = 12;                    // Fast EMA period
input int      SlowMA = 26;                    // Slow EMA period
input int      SignalMA = 9;                   // Signal MA period
input int      RSI_Period = 14;                // RSI period
input int      RSI_Overbought = 70;            // RSI overbought level
input int      RSI_Oversold = 30;              // RSI oversold level
input int      BB_Period = 20;                 // Bollinger Bands period
input double   BB_Deviation = 2.0;             // Bollinger Bands deviation
input int      ATR_Period = 14;                // ATR period

input group "=== ML Signal Settings ==="
input double   LongThreshold = 0.60;           // Long signal threshold (0-1)
input double   ShortThreshold = 0.60;          // Short signal threshold (0-1)
input int      MinBarsForSignal = 50;          // Minimum bars required

input group "=== Time Filters ==="
input bool     UseTimeFilter = true;           // Use trading time filter
input int      StartHour = 9;                  // Trading start hour (broker time)
input int      EndHour = 16;                   // Trading end hour (broker time)
input bool     TradeMonday = true;             // Trade on Monday
input bool     TradeTuesday = true;            // Trade on Tuesday
input bool     TradeWednesday = true;          // Trade on Wednesday
input bool     TradeThursday = true;           // Trade on Thursday
input bool     TradeFriday = true;             // Trade on Friday

input group "=== Display Settings ==="
input bool     ShowPanel = true;               // Show info panel
input int      MagicNumber = 123456;           // Magic number for trades
input string   CommentPrefix = "ML_EA";        // Order comment prefix

//--- Global variables
double point;
int digits;
double tickValue;
datetime lastBarTime;

// Indicator handles
int handleEMA_Fast;
int handleEMA_Slow;
int handleRSI;
int handleBB;
int handleATR;

// Arrays for indicator values
double emaFast[];
double emaSlow[];
double rsi[];
double bbUpper[];
double bbMiddle[];
double bbLower[];
double atr[];

// ML feature arrays
double features[50];  // Store calculated features for ML

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("=== ML Trading EA Initialized ===");
   Print("Symbol: ", _Symbol);
   Print("Timeframe: ", EnumToString(_Period));

   // Get symbol properties
   point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);

   // Initialize indicators
   handleEMA_Fast = iMA(_Symbol, _Period, FastMA, 0, MODE_EMA, PRICE_CLOSE);
   handleEMA_Slow = iMA(_Symbol, _Period, SlowMA, 0, MODE_EMA, PRICE_CLOSE);
   handleRSI = iRSI(_Symbol, _Period, RSI_Period, PRICE_CLOSE);
   handleBB = iBands(_Symbol, _Period, BB_Period, 0, BB_Deviation, PRICE_CLOSE);
   handleATR = iATR(_Symbol, _Period, ATR_Period);

   // Check if indicators initialized
   if(handleEMA_Fast == INVALID_HANDLE || handleEMA_Slow == INVALID_HANDLE ||
      handleRSI == INVALID_HANDLE || handleBB == INVALID_HANDLE ||
      handleATR == INVALID_HANDLE)
   {
      Print("ERROR: Failed to initialize indicators!");
      return(INIT_FAILED);
   }

   // Set array as series
   ArraySetAsSeries(emaFast, true);
   ArraySetAsSeries(emaSlow, true);
   ArraySetAsSeries(rsi, true);
   ArraySetAsSeries(bbUpper, true);
   ArraySetAsSeries(bbMiddle, true);
   ArraySetAsSeries(bbLower, true);
   ArraySetAsSeries(atr, true);

   lastBarTime = 0;

   Print("Initialization successful!");
   Print("Max Spread: ", MaxSpreadPips, " pips");
   Print("Stop Loss: ", StopLossPips, " pips | Take Profit: ", TakeProfitPips, " pips");

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("EA deinitialized. Reason: ", reason);

   // Release indicator handles
   IndicatorRelease(handleEMA_Fast);
   IndicatorRelease(handleEMA_Slow);
   IndicatorRelease(handleRSI);
   IndicatorRelease(handleBB);
   IndicatorRelease(handleATR);

   // Remove panel objects
   ObjectsDeleteAll(0, CommentPrefix);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check if new bar
   datetime currentBarTime = iTime(_Symbol, _Period, 0);
   if(currentBarTime == lastBarTime)
      return;  // Not a new bar

   lastBarTime = currentBarTime;

   // Check minimum bars
   if(Bars(_Symbol, _Period) < MinBarsForSignal)
   {
      Print("Not enough bars. Need: ", MinBarsForSignal);
      return;
   }

   // Check time filter
   if(UseTimeFilter && !IsTimeToTrade())
      return;

   // Check spread
   if(!IsSpreadAcceptable())
   {
      Print("Spread too high: ", GetCurrentSpread(), " pips");
      return;
   }

   // Update indicators
   if(!UpdateIndicators())
   {
      Print("Failed to update indicators");
      return;
   }

   // Calculate features for ML model
   CalculateFeatures();

   // Generate ML signals
   double longScore = CalculateLongScore();
   double shortScore = CalculateShortScore();

   // Get current position
   int currentPositions = CountPositions();

   // Show info panel
   if(ShowPanel)
      DisplayInfoPanel(longScore, shortScore, currentPositions);

   // Check for trailing stop
   if(UseTrailingStop && currentPositions > 0)
      ManageTrailingStop();

   // Trading logic
   if(currentPositions >= MaxPositions)
      return;  // Max positions reached

   // LONG signal
   if(AllowLong && longScore > LongThreshold && shortScore < 0.5)
   {
      Print("LONG SIGNAL! Score: ", DoubleToString(longScore, 2));
      OpenPosition(ORDER_TYPE_BUY, longScore);
   }
   // SHORT signal
   else if(AllowShort && shortScore > ShortThreshold && longScore < 0.5)
   {
      Print("SHORT SIGNAL! Score: ", DoubleToString(shortScore, 2));
      OpenPosition(ORDER_TYPE_SELL, shortScore);
   }
}

//+------------------------------------------------------------------+
//| Update all indicator buffers                                      |
//+------------------------------------------------------------------+
bool UpdateIndicators()
{
   if(CopyBuffer(handleEMA_Fast, 0, 0, 50, emaFast) <= 0) return false;
   if(CopyBuffer(handleEMA_Slow, 0, 0, 50, emaSlow) <= 0) return false;
   if(CopyBuffer(handleRSI, 0, 0, 50, rsi) <= 0) return false;
   if(CopyBuffer(handleBB, 0, 0, 50, bbUpper) <= 0) return false;
   if(CopyBuffer(handleBB, 1, 0, 50, bbMiddle) <= 0) return false;
   if(CopyBuffer(handleBB, 2, 0, 50, bbLower) <= 0) return false;
   if(CopyBuffer(handleATR, 0, 0, 50, atr) <= 0) return false;

   return true;
}

//+------------------------------------------------------------------+
//| Calculate features for ML model (similar to Python version)      |
//+------------------------------------------------------------------+
void CalculateFeatures()
{
   // Get OHLC data
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int copied = CopyRates(_Symbol, _Period, 0, 50, rates);

   if(copied < 50) return;

   int idx = 0;

   // Moving Average features
   features[idx++] = (rates[0].close - emaFast[0]) / rates[0].close;      // Distance to fast MA
   features[idx++] = (rates[0].close - emaSlow[0]) / rates[0].close;      // Distance to slow MA
   features[idx++] = (emaFast[0] > emaSlow[0]) ? 1.0 : 0.0;               // MA trend

   // MACD
   double macd = emaFast[0] - emaSlow[0];
   features[idx++] = macd / rates[0].close;                                // MACD normalized

   // RSI features
   features[idx++] = rsi[0] / 100.0;                                       // RSI normalized
   features[idx++] = (rsi[0] > RSI_Overbought) ? 1.0 : 0.0;              // Overbought
   features[idx++] = (rsi[0] < RSI_Oversold) ? 1.0 : 0.0;                // Oversold

   // Bollinger Bands
   features[idx++] = (rates[0].close - bbLower[0]) / (bbUpper[0] - bbLower[0]);  // BB position
   double bbWidth = (bbUpper[0] - bbLower[0]) / bbMiddle[0];
   features[idx++] = bbWidth;                                              // BB width

   // ATR (volatility)
   features[idx++] = atr[0] / rates[0].close;                             // ATR normalized

   // Price momentum
   if(copied > 5)
   {
      features[idx++] = (rates[0].close - rates[5].close) / rates[5].close;   // 5-bar momentum
      features[idx++] = (rates[0].close - rates[10].close) / rates[10].close; // 10-bar momentum
   }

   // Volume (if available)
   features[idx++] = (double)rates[0].tick_volume;

   // Price patterns
   features[idx++] = (rates[0].close > rates[1].close) ? 1.0 : 0.0;      // Higher close
   features[idx++] = (rates[0].high > rates[1].high) ? 1.0 : 0.0;        // Higher high
   features[idx++] = (rates[0].low < rates[1].low) ? 1.0 : 0.0;          // Lower low

   // Candle body/shadow analysis
   double body = MathAbs(rates[0].close - rates[0].open);
   double range = rates[0].high - rates[0].low;
   features[idx++] = (range > 0) ? body / range : 0.5;                    // Body ratio
}

//+------------------------------------------------------------------+
//| Calculate LONG signal score (0-1) - ML ensemble simulation       |
//+------------------------------------------------------------------+
double CalculateLongScore()
{
   double score = 0.5;  // Start neutral
   int signals = 0;

   // MA crossover (bullish)
   if(emaFast[0] > emaSlow[0] && emaFast[1] <= emaSlow[1])
   {
      score += 0.15;
      signals++;
   }

   // Price above MA
   if(iClose(_Symbol, _Period, 0) > emaFast[0])
      score += 0.05;

   // RSI oversold bounce
   if(rsi[0] > RSI_Oversold && rsi[1] <= RSI_Oversold)
   {
      score += 0.10;
      signals++;
   }

   // RSI bullish (30-70 range)
   if(rsi[0] > 40 && rsi[0] < 60)
      score += 0.05;

   // BB bounce from lower band
   double close = iClose(_Symbol, _Period, 0);
   if(close > bbLower[0] && iClose(_Symbol, _Period, 1) <= bbLower[1])
   {
      score += 0.10;
      signals++;
   }

   // Price in lower half of BB (buy zone)
   double bbPosition = (close - bbLower[0]) / (bbUpper[0] - bbLower[0]);
   if(bbPosition < 0.3)
      score += 0.08;

   // Momentum
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   CopyRates(_Symbol, _Period, 0, 10, rates);

   if(rates[0].close > rates[5].close)  // 5-bar bullish momentum
      score += 0.07;

   // Volatility expansion (good for entry)
   if(atr[0] > atr[5])
      score += 0.05;

   // MACD bullish
   double macd = emaFast[0] - emaSlow[0];
   if(macd > 0)
      score += 0.05;

   // Bullish candle pattern
   if(rates[0].close > rates[0].open)
      score += 0.03;

   // Multiple timeframe confirmation (if applicable)
   if(_Period == PERIOD_M5)
   {
      double emaFast_H1 = iMA(_Symbol, PERIOD_H1, FastMA, 0, MODE_EMA, PRICE_CLOSE);
      double emaSlow_H1 = iMA(_Symbol, PERIOD_H1, SlowMA, 0, MODE_EMA, PRICE_CLOSE);

      double maFast[];
      double maSlow[];
      ArraySetAsSeries(maFast, true);
      ArraySetAsSeries(maSlow, true);

      if(CopyBuffer(emaFast_H1, 0, 0, 1, maFast) > 0 &&
         CopyBuffer(emaSlow_H1, 0, 0, 1, maSlow) > 0)
      {
         if(maFast[0] > maSlow[0])  // Higher TF trend confirmation
            score += 0.10;
      }

      IndicatorRelease(emaFast_H1);
      IndicatorRelease(emaSlow_H1);
   }

   // Normalize score
   score = MathMax(0.0, MathMin(1.0, score));

   return score;
}

//+------------------------------------------------------------------+
//| Calculate SHORT signal score (0-1) - ML ensemble simulation      |
//+------------------------------------------------------------------+
double CalculateShortScore()
{
   double score = 0.5;  // Start neutral
   int signals = 0;

   // MA crossover (bearish)
   if(emaFast[0] < emaSlow[0] && emaFast[1] >= emaSlow[1])
   {
      score += 0.15;
      signals++;
   }

   // Price below MA
   if(iClose(_Symbol, _Period, 0) < emaFast[0])
      score += 0.05;

   // RSI overbought reversal
   if(rsi[0] < RSI_Overbought && rsi[1] >= RSI_Overbought)
   {
      score += 0.10;
      signals++;
   }

   // RSI bearish (30-70 range)
   if(rsi[0] > 40 && rsi[0] < 60)
      score += 0.05;

   // BB rejection from upper band
   double close = iClose(_Symbol, _Period, 0);
   if(close < bbUpper[0] && iClose(_Symbol, _Period, 1) >= bbUpper[1])
   {
      score += 0.10;
      signals++;
   }

   // Price in upper half of BB (sell zone)
   double bbPosition = (close - bbLower[0]) / (bbUpper[0] - bbLower[0]);
   if(bbPosition > 0.7)
      score += 0.08;

   // Momentum
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   CopyRates(_Symbol, _Period, 0, 10, rates);

   if(rates[0].close < rates[5].close)  // 5-bar bearish momentum
      score += 0.07;

   // Volatility expansion
   if(atr[0] > atr[5])
      score += 0.05;

   // MACD bearish
   double macd = emaFast[0] - emaSlow[0];
   if(macd < 0)
      score += 0.05;

   // Bearish candle pattern
   if(rates[0].close < rates[0].open)
      score += 0.03;

   // Multiple timeframe confirmation
   if(_Period == PERIOD_M5)
   {
      int emaFast_H1 = iMA(_Symbol, PERIOD_H1, FastMA, 0, MODE_EMA, PRICE_CLOSE);
      int emaSlow_H1 = iMA(_Symbol, PERIOD_H1, SlowMA, 0, MODE_EMA, PRICE_CLOSE);

      double maFast[];
      double maSlow[];
      ArraySetAsSeries(maFast, true);
      ArraySetAsSeries(maSlow, true);

      if(CopyBuffer(emaFast_H1, 0, 0, 1, maFast) > 0 &&
         CopyBuffer(emaSlow_H1, 0, 0, 1, maSlow) > 0)
      {
         if(maFast[0] < maSlow[0])  // Higher TF trend confirmation
            score += 0.10;
      }

      IndicatorRelease(emaFast_H1);
      IndicatorRelease(emaSlow_H1);
   }

   // Normalize score
   score = MathMax(0.0, MathMin(1.0, score));

   return score;
}

//+------------------------------------------------------------------+
//| Open a new position                                              |
//+------------------------------------------------------------------+
void OpenPosition(ENUM_ORDER_TYPE orderType, double signalStrength)
{
   double price, sl, tp;
   double currentSpread = GetCurrentSpread();

   // Calculate entry price
   if(orderType == ORDER_TYPE_BUY)
   {
      price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      sl = price - StopLossPips * point * 10;
      tp = price + TakeProfitPips * point * 10;
   }
   else  // SELL
   {
      price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      sl = price + StopLossPips * point * 10;
      tp = price - TakeProfitPips * point * 10;
   }

   // Normalize prices
   sl = NormalizeDouble(sl, digits);
   tp = NormalizeDouble(tp, digits);

   // Calculate lot size based on risk
   double lotSize = CalculateLotSize(StopLossPips);
   lotSize = NormalizeLot(lotSize);

   // Create trade request
   MqlTradeRequest request = {};
   MqlTradeResult result = {};

   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = lotSize;
   request.type = orderType;
   request.price = price;
   request.sl = sl;
   request.tp = tp;
   request.deviation = 10;
   request.magic = MagicNumber;
   request.comment = CommentPrefix + "_Score:" + DoubleToString(signalStrength, 2);
   request.type_filling = ORDER_FILLING_FOK;

   // Send order
   if(OrderSend(request, result))
   {
      if(result.retcode == TRADE_RETCODE_DONE)
      {
         Print("✓ Position opened: ", EnumToString(orderType),
               " | Lot: ", lotSize,
               " | Price: ", price,
               " | SL: ", sl,
               " | TP: ", tp,
               " | Score: ", DoubleToString(signalStrength, 2));
      }
      else
      {
         Print("✗ Order failed! Retcode: ", result.retcode, " - ", result.comment);
      }
   }
   else
   {
      Print("✗ OrderSend failed! Error: ", GetLastError());
   }
}

//+------------------------------------------------------------------+
//| Calculate lot size based on risk percentage                      |
//+------------------------------------------------------------------+
double CalculateLotSize(double stopLossPips)
{
   double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = accountBalance * (RiskPercent / 100.0);

   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double pointValue = tickValue * (point / tickSize);

   double stopLossValue = stopLossPips * 10 * pointValue;

   double lotSize = LotSize;
   if(stopLossValue > 0)
      lotSize = riskAmount / stopLossValue;

   return lotSize;
}

//+------------------------------------------------------------------+
//| Normalize lot size                                               |
//+------------------------------------------------------------------+
double NormalizeLot(double lots)
{
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   lots = MathMax(minLot, MathMin(maxLot, lots));
   lots = MathFloor(lots / lotStep) * lotStep;

   return NormalizeDouble(lots, 2);
}

//+------------------------------------------------------------------+
//| Count open positions                                             |
//+------------------------------------------------------------------+
int CountPositions()
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket))
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
            PositionGetInteger(POSITION_MAGIC) == MagicNumber)
         {
            count++;
         }
      }
   }
   return count;
}

//+------------------------------------------------------------------+
//| Manage trailing stop for open positions                          |
//+------------------------------------------------------------------+
void ManageTrailingStop()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;

      if(PositionGetString(POSITION_SYMBOL) != _Symbol ||
         PositionGetInteger(POSITION_MAGIC) != MagicNumber)
         continue;

      double positionOpenPrice = PositionGetDouble(POSITION_PRICE_OPEN);
      double currentSL = PositionGetDouble(POSITION_SL);
      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      double newSL = 0;
      bool modifyNeeded = false;

      if(posType == POSITION_TYPE_BUY)
      {
         double currentBid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         newSL = currentBid - TrailingStopPips * point * 10;
         newSL = NormalizeDouble(newSL, digits);

         if(newSL > currentSL && newSL < currentBid)
            modifyNeeded = true;
      }
      else  // SELL
      {
         double currentAsk = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         newSL = currentAsk + TrailingStopPips * point * 10;
         newSL = NormalizeDouble(newSL, digits);

         if((currentSL == 0 || newSL < currentSL) && newSL > currentAsk)
            modifyNeeded = true;
      }

      if(modifyNeeded)
      {
         MqlTradeRequest request = {};
         MqlTradeResult result = {};

         request.action = TRADE_ACTION_SLTP;
         request.position = ticket;
         request.symbol = _Symbol;
         request.sl = newSL;
         request.tp = PositionGetDouble(POSITION_TP);

         if(OrderSend(request, result))
         {
            if(result.retcode == TRADE_RETCODE_DONE)
               Print("Trailing stop updated for ticket ", ticket, " New SL: ", newSL);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Check if current time is within trading hours                    |
//+------------------------------------------------------------------+
bool IsTimeToTrade()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);

   // Check day of week
   bool dayAllowed = false;
   switch(dt.day_of_week)
   {
      case 1: dayAllowed = TradeMonday; break;
      case 2: dayAllowed = TradeTuesday; break;
      case 3: dayAllowed = TradeWednesday; break;
      case 4: dayAllowed = TradeThursday; break;
      case 5: dayAllowed = TradeFriday; break;
      default: dayAllowed = false;
   }

   if(!dayAllowed) return false;

   // Check hour
   if(dt.hour < StartHour || dt.hour >= EndHour)
      return false;

   return true;
}

//+------------------------------------------------------------------+
//| Check if spread is acceptable                                    |
//+------------------------------------------------------------------+
bool IsSpreadAcceptable()
{
   double spread = GetCurrentSpread();
   return (spread <= MaxSpreadPips);
}

//+------------------------------------------------------------------+
//| Get current spread in pips                                       |
//+------------------------------------------------------------------+
double GetCurrentSpread()
{
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   return (ask - bid) / (point * 10);
}

//+------------------------------------------------------------------+
//| Display information panel on chart                               |
//+------------------------------------------------------------------+
void DisplayInfoPanel(double longScore, double shortScore, int positions)
{
   string panelText = "\n";
   panelText += "═══════════════════════════════════\n";
   panelText += "   ML TRADING EA - " + _Symbol + "\n";
   panelText += "═══════════════════════════════════\n";
   panelText += "Time: " + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS) + "\n";
   panelText += "Spread: " + DoubleToString(GetCurrentSpread(), 1) + " pips\n";
   panelText += "───────────────────────────────────\n";
   panelText += "LONG Score:  " + DoubleToString(longScore, 2) + " ";
   panelText += (longScore > LongThreshold) ? "✓ SIGNAL\n" : "\n";
   panelText += "SHORT Score: " + DoubleToString(shortScore, 2) + " ";
   panelText += (shortScore > ShortThreshold) ? "✓ SIGNAL\n" : "\n";
   panelText += "───────────────────────────────────\n";
   panelText += "Open Positions: " + IntegerToString(positions) + "/" + IntegerToString(MaxPositions) + "\n";
   panelText += "RSI: " + DoubleToString(rsi[0], 1) + "\n";
   panelText += "ATR: " + DoubleToString(atr[0] / point, 1) + " pips\n";
   panelText += "═══════════════════════════════════\n";

   Comment(panelText);
}
//+------------------------------------------------------------------+
