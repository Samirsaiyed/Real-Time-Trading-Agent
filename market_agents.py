from langchain_openai import ChatOpenAI
from typing import Dict, Any, List, Optional
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import json
from dotenv import load_dotenv

load_dotenv()

class MarketAnalysisAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        self.supported_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "SPY", "QQQ"]
    
    async def analyze_market_data(self, symbols: List[str], timeframe: str = "1d") -> Dict[str, Any]:
        print(f"📊 Analyzing market data for {len(symbols)} symbols...")
        
        # Fetch real market data
        market_data = await self._fetch_market_data(symbols, timeframe)
        
        # Technical analysis
        technical_signals = await self._perform_technical_analysis(market_data)
        
        # Market sentiment analysis
        sentiment_analysis = await self._analyze_market_sentiment(symbols, market_data)
        
        # Generate market overview
        market_overview = await self._generate_market_overview(market_data, technical_signals, sentiment_analysis)
        
        return {
            "symbols": symbols,
            "timeframe": timeframe,
            "market_data": market_data,
            "technical_signals": technical_signals,
            "sentiment": sentiment_analysis,
            "overview": market_overview,
            "timestamp": datetime.now().isoformat(),
            "agent": "MarketAnalysisAgent"
        }
    
    async def _fetch_market_data(self, symbols: List[str], timeframe: str) -> Dict[str, pd.DataFrame]:
        market_data = {}
        
        for symbol in symbols:
            try:
                # Fetch data from yfinance
                ticker = yf.Ticker(symbol)
                
                # Get historical data based on timeframe
                if timeframe == "1d":
                    data = ticker.history(period="30d", interval="1h")
                elif timeframe == "1h":
                    data = ticker.history(period="7d", interval="15m")
                else:
                    data = ticker.history(period="90d", interval="1d")
                
                # Calculate technical indicators
                data = self._add_technical_indicators(data)
                market_data[symbol] = data
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                market_data[symbol] = pd.DataFrame()
        
        return market_data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators to market data"""
        if len(data) < 20:
            return data
        
        # Moving averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=min(50, len(data))).mean()
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        return data
    
    async def _perform_technical_analysis(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        technical_signals = {}
        
        for symbol, data in market_data.items():
            if data.empty or len(data) < 20:
                technical_signals[symbol] = {"signal": "INSUFFICIENT_DATA", "strength": 0}
                continue
            
            latest = data.iloc[-1]
            previous = data.iloc[-2] if len(data) > 1 else latest
            
            signals = []
            strength_scores = []
            
            # Moving Average Signals
            if latest['Close'] > latest['SMA_20']:
                signals.append("MA_BULLISH")
                strength_scores.append(0.3)
            elif latest['Close'] < latest['SMA_20']:
                signals.append("MA_BEARISH")
                strength_scores.append(-0.3)
            
            # RSI Signals
            if not pd.isna(latest['RSI']):
                if latest['RSI'] > 70:
                    signals.append("RSI_OVERBOUGHT")
                    strength_scores.append(-0.4)
                elif latest['RSI'] < 30:
                    signals.append("RSI_OVERSOLD")
                    strength_scores.append(0.4)
            
            # MACD Signals
            if not pd.isna(latest['MACD']) and not pd.isna(previous['MACD']):
                if latest['MACD'] > latest['MACD_Signal'] and previous['MACD'] <= previous['MACD_Signal']:
                    signals.append("MACD_BULLISH_CROSS")
                    strength_scores.append(0.5)
                elif latest['MACD'] < latest['MACD_Signal'] and previous['MACD'] >= previous['MACD_Signal']:
                    signals.append("MACD_BEARISH_CROSS")
                    strength_scores.append(-0.5)
            
            # Volume Analysis
            if latest['Volume_Ratio'] > 1.5:
                signals.append("HIGH_VOLUME")
                strength_scores.append(0.2)
            
            # Overall signal calculation
            total_strength = sum(strength_scores)
            
            if total_strength > 0.3:
                overall_signal = "BUY"
            elif total_strength < -0.3:
                overall_signal = "SELL"
            else:
                overall_signal = "HOLD"
            
            technical_signals[symbol] = {
                "signal": overall_signal,
                "strength": total_strength,
                "individual_signals": signals,
                "price": float(latest['Close']),
                "change_pct": ((latest['Close'] - previous['Close']) / previous['Close']) * 100,
                "volume_ratio": float(latest.get('Volume_Ratio', 1.0)),
                "rsi": float(latest.get('RSI', 50.0)) if not pd.isna(latest.get('RSI', np.nan)) else 50.0
            }
        
        return technical_signals
    
    async def _analyze_market_sentiment(self, symbols: List[str], market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        # Create market context for LLM analysis
        market_summary = []
        
        for symbol, data in market_data.items():
            if not data.empty:
                latest = data.iloc[-1]
                prev_close = data.iloc[-2]['Close'] if len(data) > 1 else latest['Close']
                change_pct = ((latest['Close'] - prev_close) / prev_close) * 100
                
                market_summary.append(f"{symbol}: ${latest['Close']:.2f} ({change_pct:+.2f}%)")
        
        market_context = "\n".join(market_summary)
        
        prompt = f"""Analyze the current market sentiment based on this data:

        {market_context}

        Consider:
        1. Overall market direction and momentum
        2. Sector performance patterns
        3. Risk-on vs risk-off sentiment
        4. Key market themes and catalysts
        
        Provide:
        - Overall sentiment (BULLISH/BEARISH/NEUTRAL)
        - Confidence level (0-100)
        - Key market themes (3-4 points)
        - Risk factors to watch
        
        Be concise and data-driven."""
        
        response = await asyncio.to_thread(
            self.llm.invoke, [{"role": "user", "content": prompt}]
        )
        
        # Parse LLM response (simplified parsing)
        sentiment_text = response.content
        
        # Extract sentiment (basic parsing - in production, you'd use more sophisticated NLP)
        if "BULLISH" in sentiment_text.upper():
            overall_sentiment = "BULLISH"
        elif "BEARISH" in sentiment_text.upper():
            overall_sentiment = "BEARISH"
        else:
            overall_sentiment = "NEUTRAL"
        
        return {
            "overall_sentiment": overall_sentiment,
            "analysis": sentiment_text,
            "market_summary": market_context,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _generate_market_overview(self, market_data: Dict[str, pd.DataFrame], 
                                      technical_signals: Dict[str, Dict[str, Any]], 
                                      sentiment: Dict[str, Any]) -> str:
        
        # Create comprehensive market overview
        signal_summary = []
        for symbol, signal_data in technical_signals.items():
            signal_summary.append(f"{symbol}: {signal_data['signal']} (strength: {signal_data['strength']:.2f})")
        
        signals_text = "\n".join(signal_summary)
        
        prompt = f"""Create a professional market overview based on this analysis:

        TECHNICAL SIGNALS:
        {signals_text}
        
        SENTIMENT ANALYSIS:
        {sentiment['analysis']}
        
        Generate a concise 3-4 sentence market overview that:
        1. Summarizes the current market state
        2. Highlights key opportunities or risks
        3. Provides actionable insights for traders
        
        Keep it professional and data-driven."""
        
        response = await asyncio.to_thread(
            self.llm.invoke, [{"role": "user", "content": prompt}]
        )
        
        return response.content

class RiskManagementAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.05)  # Very low temperature for risk calculations
        self.max_position_size = 0.1  # 10% max position size
        self.max_portfolio_risk = 0.02  # 2% max portfolio risk per trade
    
    async def assess_portfolio_risk(self, portfolio: Dict[str, Any], market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        print("⚠️ Assessing portfolio risk...")
        
        # Calculate current risk metrics
        risk_metrics = await self._calculate_risk_metrics(portfolio, market_analysis)
        
        # Generate risk recommendations
        recommendations = await self._generate_risk_recommendations(risk_metrics, portfolio)
        
        # Calculate position sizing recommendations
        position_sizing = await self._calculate_position_sizing(portfolio, market_analysis)
        
        return {
            "portfolio_value": portfolio.get("total_value", 100000),
            "current_risk": risk_metrics,
            "recommendations": recommendations,
            "position_sizing": position_sizing,
            "risk_score": risk_metrics.get("overall_risk_score", 5),  # 1-10 scale
            "timestamp": datetime.now().isoformat(),
            "agent": "RiskManagementAgent"
        }
    
    async def _calculate_risk_metrics(self, portfolio: Dict[str, Any], market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        # Simplified risk calculations (in production, you'd use more sophisticated models)
        
        total_value = portfolio.get("total_value", 100000)
        positions = portfolio.get("positions", {})
        
        # Calculate portfolio concentration
        position_values = list(positions.values()) if positions else [total_value]
        max_position = max(position_values) if position_values else 0
        concentration_risk = (max_position / total_value) * 10  # Scale to 1-10
        
        # Calculate volatility risk based on technical signals
        technical_signals = market_analysis.get("technical_signals", {})
        volatility_scores = []
        
        for symbol, signal_data in technical_signals.items():
            # Use RSI and volume ratio as volatility proxies
            rsi = signal_data.get("rsi", 50)
            volume_ratio = signal_data.get("volume_ratio", 1.0)
            
            # Higher RSI extremes and volume spikes indicate higher volatility
            vol_score = abs(rsi - 50) / 10 + (volume_ratio - 1) * 2
            volatility_scores.append(min(vol_score, 10))  # Cap at 10
        
        avg_volatility = np.mean(volatility_scores) if volatility_scores else 5
        
        # Overall risk score (1-10, where 10 is highest risk)
        overall_risk = (concentration_risk * 0.4 + avg_volatility * 0.6)
        overall_risk = min(max(overall_risk, 1), 10)  # Clamp to 1-10 range
        
        return {
            "concentration_risk": round(concentration_risk, 2),
            "volatility_risk": round(avg_volatility, 2),
            "overall_risk_score": round(overall_risk, 2),
            "portfolio_diversification": len(positions) if positions else 0,
            "max_position_percentage": round((max_position / total_value) * 100, 2) if total_value > 0 else 0
        }
    
    async def _generate_risk_recommendations(self, risk_metrics: Dict[str, Any], portfolio: Dict[str, Any]) -> List[str]:
        recommendations = []
        
        # Concentration risk recommendations  
        if risk_metrics["concentration_risk"] > 7:
            recommendations.append("⚠️ HIGH CONCENTRATION RISK: Consider diversifying positions")
        
        # Volatility risk recommendations
        if risk_metrics["volatility_risk"] > 7:
            recommendations.append("📈 HIGH VOLATILITY: Consider reducing position sizes")
        
        # Diversification recommendations
        if risk_metrics["portfolio_diversification"] < 3:
            recommendations.append("🎯 LOW DIVERSIFICATION: Add positions in different sectors")
        
        # Overall risk recommendations
        if risk_metrics["overall_risk_score"] > 8:
            recommendations.append("🔴 CRITICAL RISK LEVEL: Immediate position review required")
        elif risk_metrics["overall_risk_score"] > 6:
            recommendations.append("🟡 ELEVATED RISK: Monitor positions closely")
        else:
            recommendations.append("🟢 ACCEPTABLE RISK: Current risk levels are manageable")
        
        return recommendations
    
    async def _calculate_position_sizing(self, portfolio: Dict[str, Any], market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        total_value = portfolio.get("total_value", 100000)
        technical_signals = market_analysis.get("technical_signals", {})
        
        position_recommendations = {}
        
        for symbol, signal_data in technical_signals.items():
            signal_strength = abs(signal_data.get("strength", 0))
            risk_score = abs(signal_data.get("rsi", 50) - 50) / 50  # Normalize RSI to risk score
            
            # Calculate recommended position size based on signal strength and risk
            base_size = min(self.max_position_size, signal_strength * 0.15)  # Max 15% for strongest signals
            risk_adjusted_size = base_size * (1 - risk_score * 0.5)  # Reduce size for higher risk
            
            recommended_value = total_value * risk_adjusted_size
            
            position_recommendations[symbol] = {
                "recommended_percentage": round(risk_adjusted_size * 100, 2),
                "recommended_value": round(recommended_value, 2),
                "signal_strength": round(signal_strength, 2),
                "risk_adjustment": round(risk_score, 2),
                "max_loss": round(recommended_value * self.max_portfolio_risk, 2)
            }
        
        return position_recommendations

# Test the advanced agents
if __name__ == "__main__":
    async def test_agents():
        market_agent = MarketAnalysisAgent()
        risk_agent = RiskManagementAgent()
        
        print("🚀 Testing advanced trading agents...")
        
        # Test market analysis
        symbols = ["AAPL", "GOOGL", "TSLA"]
        market_analysis = await market_agent.analyze_market_data(symbols)
        
        print(f"✅ Market analysis complete for {len(symbols)} symbols")
        print(f"📊 Technical signals generated")
        print(f"💭 Sentiment analysis: {market_analysis['sentiment']['overall_sentiment']}")
        
        # Test risk management
        sample_portfolio = {
            "total_value": 100000,
            "positions": {
                "AAPL": 30000,
                "GOOGL": 25000,
                "TSLA": 20000
            }
        }
        
        risk_assessment = await risk_agent.assess_portfolio_risk(sample_portfolio, market_analysis)
        
        print(f"⚠️ Risk assessment complete")
        print(f"📊 Overall risk score: {risk_assessment['risk_score']}/10")
        print(f"💡 Recommendations: {len(risk_assessment['recommendations'])}")
        
        print("\n🔥 Advanced trading agents are ready!")
        
    # Run the async test
    asyncio.run(test_agents())