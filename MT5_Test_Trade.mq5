//+------------------------------------------------------------------+
//|                                            MT5_Test_Trade.mq5    |
//|                              Simple script to test trade execution|
//|                              Opens 1 BUY trade immediately        |
//+------------------------------------------------------------------+
#property copyright "Test Script"
#property version   "1.00"
#property script_show_inputs

//--- Input parameters
input ENUM_ORDER_TYPE OrderType = ORDER_TYPE_BUY;  // Order Type (BUY or SELL)
input double LotSize = 0.01;                        // Lot Size
input int StopLossPips = 20;                        // Stop Loss in Pips
input int TakeProfitPips = 30;                      // Take Profit in Pips
input string TestComment = "TEST_TRADE";            // Comment

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
{
   Print("========================================");
   Print("STARTING TEST TRADE");
   Print("========================================");

   string symbol = _Symbol;
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);

   // Get current prices
   double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(symbol, SYMBOL_BID);

   Print("Symbol: ", symbol);
   Print("Ask: ", ask, " | Bid: ", bid);
   Print("Lot Size: ", LotSize);

   // Calculate SL and TP
   double price, sl, tp;

   if(OrderType == ORDER_TYPE_BUY)
   {
      price = ask;
      sl = price - StopLossPips * point * 10;
      tp = price + TakeProfitPips * point * 10;
      Print("Opening BUY position...");
   }
   else
   {
      price = bid;
      sl = price + StopLossPips * point * 10;
      tp = price - TakeProfitPips * point * 10;
      Print("Opening SELL position...");
   }

   // Normalize
   sl = NormalizeDouble(sl, digits);
   tp = NormalizeDouble(tp, digits);

   Print("Entry: ", price);
   Print("Stop Loss: ", sl);
   Print("Take Profit: ", tp);

   // Create trade request
   MqlTradeRequest request = {};
   MqlTradeResult result = {};

   request.action = TRADE_ACTION_DEAL;
   request.symbol = symbol;
   request.volume = LotSize;
   request.type = OrderType;
   request.price = price;
   request.sl = sl;
   request.tp = tp;
   request.deviation = 10;
   request.magic = 999999;
   request.comment = TestComment;
   request.type_filling = ORDER_FILLING_FOK;

   // Send order
   Print("Sending order to broker...");

   if(!OrderSend(request, result))
   {
      Print("========================================");
      Print("ORDER FAILED!");
      Print("Error Code: ", GetLastError());
      Print("========================================");
      return;
   }

   // Check result
   if(result.retcode == TRADE_RETCODE_DONE)
   {
      Print("========================================");
      Print("✓✓✓ TRADE SUCCESSFUL! ✓✓✓");
      Print("========================================");
      Print("Order Ticket: ", result.order);
      Print("Deal Ticket: ", result.deal);
      Print("Volume: ", result.volume);
      Print("Price: ", result.price);
      Print("Bid: ", result.bid);
      Print("Ask: ", result.ask);
      Print("Comment: ", result.comment);
      Print("========================================");

      Alert("✓ TEST TRADE OPENED! Ticket: ", result.order);
   }
   else
   {
      Print("========================================");
      Print("ORDER REJECTED!");
      Print("Return Code: ", result.retcode);
      Print("Comment: ", result.comment);
      Print("========================================");

      // Detailed error messages
      switch(result.retcode)
      {
         case TRADE_RETCODE_INVALID:
            Print("ERROR: Invalid request");
            break;
         case TRADE_RETCODE_NO_MONEY:
            Print("ERROR: Not enough money! Reduce lot size.");
            break;
         case TRADE_RETCODE_INVALID_PRICE:
            Print("ERROR: Invalid price");
            break;
         case TRADE_RETCODE_INVALID_STOPS:
            Print("ERROR: Invalid SL/TP. Try increasing distance.");
            break;
         case TRADE_RETCODE_MARKET_CLOSED:
            Print("ERROR: Market is closed");
            break;
         case TRADE_RETCODE_TRADE_DISABLED:
            Print("ERROR: Trading is disabled. Enable Algo Trading!");
            break;
         default:
            Print("ERROR: Unknown error code: ", result.retcode);
      }

      Alert("✗ TEST TRADE FAILED! Check Experts tab.");
   }
}
//+------------------------------------------------------------------+
