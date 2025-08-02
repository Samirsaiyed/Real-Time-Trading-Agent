from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Dict, Any, List, Optional, Annotated
from market_agents import MarketAnalysisAgent, RiskManagementAgent
import asyncio
import operator
from datetime import datetime
import json

# Advanced state with parallel processing capabilities
class TradingState(TypedDict):
    # Input parameters
    symbols: List[str]
    portfolio: Dict[str, Any]
    trading_mode: str  # "analysis", "backtest", "live"
    user_preferences: Dict[str, Any]
    
    # Parallel processing results
    market_analysis: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    strategy_signals: Dict[str, Any]
    execution_plan: Dict[str, Any]
    
    # Processing control
    processing_stage: str
    parallel_results: Annotated[List[Dict[str, Any]], operator.add]  # Accumulate parallel results
    error_log: List[str]
    confidence_scores: Dict[str, float]
    
    # Final outputs
    trading_recommendations: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    real_time_updates: List[Dict[str, Any]]

class StrategyAgent:
    def __init__(self):
        self.strategies = ["momentum", "mean_reversion", "breakout", "trend_following"]
    
    async def generate_trading_signals(self, market_analysis: Dict[str, Any], risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        print("🎯 Generating trading strategies...")
        
        technical_signals = market_analysis.get("technical_signals", {})
        position_sizing = risk_assessment.get("position_sizing", {})
        
        strategy_signals = {}
        
        for symbol in technical_signals.keys():
            signal_data = technical_signals[symbol]
            position_data = position_sizing.get(symbol, {})
            
            # Multi-strategy signal generation
            momentum_signal = self._momentum_strategy(signal_data)
            mean_reversion_signal = self._mean_reversion_strategy(signal_data)
            breakout_signal = self._breakout_strategy(signal_data)
            
            # Combine strategies with risk-adjusted weighting
            combined_signal = self._combine_strategies(
                momentum_signal, mean_reversion_signal, breakout_signal, position_data
            )
            
            strategy_signals[symbol] = combined_signal
        
        return {
            "strategy_signals": strategy_signals,
            "strategy_performance": await self._calculate_strategy_performance(strategy_signals),
            "best_opportunities": self._rank_opportunities(strategy_signals),
            "timestamp": datetime.now().isoformat(),
            "agent": "StrategyAgent"
        }
    
    def _momentum_strategy(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Momentum-based trading strategy"""
        strength = signal_data.get("strength", 0)
        rsi = signal_data.get("rsi", 50)
        
        # Momentum favors trending moves
        if strength > 0.3 and 30 < rsi < 70:
            return {"signal": "BUY", "confidence": 0.8, "strategy": "momentum"}
        elif strength < -0.3 and 30 < rsi < 70:
            return {"signal": "SELL", "confidence": 0.8, "strategy": "momentum"}
        else:
            return {"signal": "HOLD", "confidence": 0.3, "strategy": "momentum"}
    
    def _mean_reversion_strategy(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mean reversion strategy"""
        rsi = signal_data.get("rsi", 50)
        
        # Mean reversion looks for oversold/overbought conditions
        if rsi < 30:
            return {"signal": "BUY", "confidence": 0.7, "strategy": "mean_reversion"}
        elif rsi > 70:
            return {"signal": "SELL", "confidence": 0.7, "strategy": "mean_reversion"}
        else:
            return {"signal": "HOLD", "confidence": 0.2, "strategy": "mean_reversion"}
    
    def _breakout_strategy(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Breakout strategy based on volume and signals"""
        volume_ratio = signal_data.get("volume_ratio", 1.0)
        individual_signals = signal_data.get("individual_signals", [])
        
        # Breakout looks for high volume with strong signals
        if volume_ratio > 1.5 and any("BULLISH" in sig for sig in individual_signals):
            return {"signal": "BUY", "confidence": 0.9, "strategy": "breakout"}
        elif volume_ratio > 1.5 and any("BEARISH" in sig for sig in individual_signals):
            return {"signal": "SELL", "confidence": 0.9, "strategy": "breakout"}
        else:
            return {"signal": "HOLD", "confidence": 0.1, "strategy": "breakout"}
    
    def _combine_strategies(self, momentum: Dict, mean_rev: Dict, breakout: Dict, position_data: Dict) -> Dict[str, Any]:
        """Combine multiple strategy signals with risk adjustment"""
        strategies = [momentum, mean_rev, breakout]
        
        # Weight signals by confidence
        buy_weight = sum(s["confidence"] for s in strategies if s["signal"] == "BUY")
        sell_weight = sum(s["confidence"] for s in strategies if s["signal"] == "SELL")
        hold_weight = sum(s["confidence"] for s in strategies if s["signal"] == "HOLD")
        
        # Determine final signal
        if buy_weight > sell_weight and buy_weight > hold_weight:
            final_signal = "BUY"
            confidence = min(buy_weight / 3, 1.0)  # Average confidence, max 1.0
        elif sell_weight > buy_weight and sell_weight > hold_weight:
            final_signal = "SELL"
            confidence = min(sell_weight / 3, 1.0)
        else:
            final_signal = "HOLD"
            confidence = max(hold_weight / 3, 0.1)
        
        # Risk adjustment
        risk_adjustment = position_data.get("risk_adjustment", 0)
        adjusted_confidence = confidence * (1 - risk_adjustment * 0.5)
        
        return {
            "final_signal": final_signal,
            "confidence": round(adjusted_confidence, 2),
            "strategy_breakdown": {
                "momentum": momentum,
                "mean_reversion": mean_rev,
                "breakout": breakout
            },
            "risk_adjusted": True,
            "position_recommendation": position_data
        }
    
    async def _calculate_strategy_performance(self, strategy_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics for strategies"""
        total_signals = len(strategy_signals)
        if total_signals == 0:
            return {"error": "No signals to analyze"}
        
        buy_signals = sum(1 for s in strategy_signals.values() if s["final_signal"] == "BUY")
        sell_signals = sum(1 for s in strategy_signals.values() if s["final_signal"] == "SELL")
        
        avg_confidence = sum(s["confidence"] for s in strategy_signals.values()) / total_signals
        
        return {
            "total_opportunities": total_signals,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "hold_signals": total_signals - buy_signals - sell_signals,
            "average_confidence": round(avg_confidence, 2),
            "signal_distribution": {
                "buy_pct": round(buy_signals / total_signals * 100, 1),
                "sell_pct": round(sell_signals / total_signals * 100, 1)
            }
        }
    
    def _rank_opportunities(self, strategy_signals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank trading opportunities by confidence and signal strength"""
        opportunities = []
        
        for symbol, signal_data in strategy_signals.items():
            if signal_data["final_signal"] in ["BUY", "SELL"]:
                opportunities.append({
                    "symbol": symbol,
                    "signal": signal_data["final_signal"],
                    "confidence": signal_data["confidence"],
                    "position_value": signal_data["position_recommendation"].get("recommended_value", 0)
                })
        
        # Sort by confidence * position_value (opportunity score)
        opportunities.sort(
            key=lambda x: x["confidence"] * (x["position_value"] / 10000), 
            reverse=True
        )
        
        return opportunities[:5]  # Top 5 opportunities

class ExecutionAgent:
    def __init__(self):
        self.order_types = ["market", "limit", "stop_loss", "take_profit"]
    
    async def create_execution_plan(self, strategy_signals: Dict[str, Any], risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        print("⚡ Creating execution plan...")
        
        strategy_data = strategy_signals.get("strategy_signals", {})
        position_sizing = risk_assessment.get("position_sizing", {})
        
        execution_orders = []
        risk_management_orders = []
        
        for symbol, signal_data in strategy_data.items():
            if signal_data["final_signal"] in ["BUY", "SELL"]:
                # Create primary order
                primary_order = self._create_primary_order(symbol, signal_data, position_sizing.get(symbol, {}))
                execution_orders.append(primary_order)
                
                # Create risk management orders
                risk_orders = self._create_risk_management_orders(symbol, signal_data, primary_order)
                risk_management_orders.extend(risk_orders)
        
        return {
            "execution_orders": execution_orders,
            "risk_management_orders": risk_management_orders,
            "total_capital_required": sum(order["value"] for order in execution_orders if order["side"] == "BUY"),
            "execution_timeline": self._create_execution_timeline(execution_orders),
            "contingency_plans": self._create_contingency_plans(execution_orders),
            "timestamp": datetime.now().isoformat(),
            "agent": "ExecutionAgent"
        }
    
    def _create_primary_order(self, symbol: str, signal_data: Dict[str, Any], position_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create the primary trading order"""
        return {
            "symbol": symbol,
            "side": signal_data["final_signal"],
            "order_type": "market",  # For demo - would be more sophisticated in production
            "quantity": position_data.get("recommended_value", 0) / 100,  # Simplified quantity calc
            "value": position_data.get("recommended_value", 0),
            "confidence": signal_data["confidence"],
            "strategy": "multi_strategy",
            "priority": "high" if signal_data["confidence"] > 0.7 else "medium",
            "estimated_execution_time": "immediate"
        }
    
    def _create_risk_management_orders(self, symbol: str, signal_data: Dict[str, Any], primary_order: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create stop-loss and take-profit orders"""
        risk_orders = []
        
        if primary_order["side"] == "BUY":
            # Stop loss at 2% below entry
            stop_loss = {
                "symbol": symbol,
                "side": "SELL",
                "order_type": "stop_loss",
                "trigger_price": primary_order["value"] * 0.98,
                "quantity": primary_order["quantity"],
                "parent_order": primary_order,
                "purpose": "risk_management"
            }
            
            # Take profit at 4% above entry
            take_profit = {
                "symbol": symbol,
                "side": "SELL", 
                "order_type": "take_profit",
                "trigger_price": primary_order["value"] * 1.04,
                "quantity": primary_order["quantity"],
                "parent_order": primary_order,
                "purpose": "profit_taking"
            }
            
            risk_orders.extend([stop_loss, take_profit])
        
        return risk_orders
    
    def _create_execution_timeline(self, orders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a timeline for order execution"""
        high_priority = [o for o in orders if o.get("priority") == "high"]
        medium_priority = [o for o in orders if o.get("priority") == "medium"]
        
        return {
            "immediate_execution": len(high_priority),
            "delayed_execution": len(medium_priority),
            "total_orders": len(orders),
            "estimated_completion": "5-10 minutes"
        }
    
    def _create_contingency_plans(self, orders: List[Dict[str, Any]]) -> List[str]:
        """Create contingency plans for execution"""
        plans = []
        
        total_value = sum(order["value"] for order in orders if order["side"] == "BUY")
        
        if total_value > 50000:
            plans.append("Large position size - consider splitting orders")
        
        if len(orders) > 5:
            plans.append("Multiple positions - stagger execution over time")
        
        plans.append("Monitor market conditions during execution")
        plans.append("Have exit strategy ready if market moves against positions")
        
        return plans

class AdvancedTradingWorkflow:
    def __init__(self):
        self.market_agent = MarketAnalysisAgent()
        self.risk_agent = RiskManagementAgent()
        self.strategy_agent = StrategyAgent()
        self.execution_agent = ExecutionAgent()
        
        # Create main workflow
        self.main_workflow = self._create_main_workflow()
        
        # Create sub-workflows for parallel processing
        self.analysis_subgraph = self._create_analysis_subgraph()
        self.strategy_subgraph = self._create_strategy_subgraph()
    
    def _create_main_workflow(self):
        """Create the main orchestration workflow"""
        workflow = StateGraph(TradingState)
        
        # Main orchestration nodes
        workflow.add_node("initialize", self._initialize_system)
        workflow.add_node("parallel_analysis", self._run_parallel_analysis)
        workflow.add_node("strategy_generation", self._run_strategy_generation)
        workflow.add_node("execution_planning", self._run_execution_planning)
        workflow.add_node("finalize_recommendations", self._finalize_recommendations)
        workflow.add_node("handle_errors", self._handle_errors)
        
        # Define flow
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "parallel_analysis")
        workflow.add_edge("parallel_analysis", "strategy_generation")
        workflow.add_edge("strategy_generation", "execution_planning")
        workflow.add_edge("execution_planning", "finalize_recommendations")
        workflow.add_edge("finalize_recommendations", END)
        workflow.add_edge("handle_errors", END)
        
        return workflow.compile()
    
    def _create_analysis_subgraph(self):
        """Create subgraph for parallel market analysis"""
        subgraph = StateGraph(TradingState)
        
        subgraph.add_node("market_analysis", self._market_analysis_node)
        subgraph.add_node("risk_analysis", self._risk_analysis_node)
        
        subgraph.set_entry_point("market_analysis")
        subgraph.add_edge("market_analysis", "risk_analysis")
        subgraph.add_edge("risk_analysis", END)
        
        return subgraph.compile()
    
    def _create_strategy_subgraph(self):
        """Create subgraph for strategy processing"""
        subgraph = StateGraph(TradingState)
        
        subgraph.add_node("generate_signals", self._generate_signals_node)
        subgraph.add_node("validate_strategies", self._validate_strategies_node)
        
        subgraph.set_entry_point("generate_signals")
        subgraph.add_edge("generate_signals", "validate_strategies")
        subgraph.add_edge("validate_strategies", END)
        
        return subgraph.compile()
    
    async def _initialize_system(self, state: TradingState) -> TradingState:
        print("🚀 Initializing advanced trading system...")
        
        return {
            **state,
            "processing_stage": "initialized",
            "parallel_results": [],
            "error_log": [],
            "confidence_scores": {},
            "real_time_updates": [{"stage": "initialization", "timestamp": datetime.now().isoformat()}]
        }
    
    async def _run_parallel_analysis(self, state: TradingState) -> TradingState:
        print("⚡ Running parallel market and risk analysis...")
        
        try:
            # Run market analysis and risk assessment in parallel
            market_task = self.market_agent.analyze_market_data(state["symbols"])
            risk_task = self.risk_agent.assess_portfolio_risk(state["portfolio"], {})
            
            # Execute in parallel
            market_analysis, risk_assessment = await asyncio.gather(market_task, risk_task)
            
            return {
                **state,
                "market_analysis": market_analysis,
                "risk_assessment": risk_assessment,
                "processing_stage": "analysis_complete",
                "confidence_scores": {
                    "market_analysis": 0.8,
                    "risk_assessment": 0.9
                },
                "parallel_results": state["parallel_results"] + [
                    {"type": "market_analysis", "status": "complete"},
                    {"type": "risk_assessment", "status": "complete"}
                ]
            }
            
        except Exception as e:
            return {
                **state,
                "error_log": state["error_log"] + [f"Parallel analysis error: {str(e)}"],
                "processing_stage": "error"
            }
    
    async def _run_strategy_generation(self, state: TradingState) -> TradingState:
        print("🎯 Running strategy generation...")
        
        try:
            strategy_signals = await self.strategy_agent.generate_trading_signals(
                state["market_analysis"], 
                state["risk_assessment"]
            )
            
            return {
                **state,
                "strategy_signals": strategy_signals,
                "processing_stage": "strategy_complete",
                "confidence_scores": {
                    **state["confidence_scores"],
                    "strategy_generation": 0.85
                }
            }
            
        except Exception as e:
            return {
                **state,
                "error_log": state["error_log"] + [f"Strategy generation error: {str(e)}"],
                "processing_stage": "error"
            }
    
    async def _run_execution_planning(self, state: TradingState) -> TradingState:
        print("⚡ Creating execution plan...")
        
        try:
            execution_plan = await self.execution_agent.create_execution_plan(
                state["strategy_signals"],
                state["risk_assessment"]
            )
            
            return {
                **state,
                "execution_plan": execution_plan,
                "processing_stage": "execution_planned",
                "confidence_scores": {
                    **state["confidence_scores"],
                    "execution_planning": 0.9
                }
            }
            
        except Exception as e:
            return {
                **state,
                "error_log": state["error_log"] + [f"Execution planning error: {str(e)}"],
                "processing_stage": "error"
            }
    
    async def _finalize_recommendations(self, state: TradingState) -> TradingState:
        print("✅ Finalizing trading recommendations...")
        
        # Compile final recommendations
        trading_recommendations = {
            "market_overview": state["market_analysis"]["overview"],
            "top_opportunities": state["strategy_signals"]["best_opportunities"],
            "execution_orders": state["execution_plan"]["execution_orders"],
            "risk_metrics": state["risk_assessment"]["current_risk"],
            "overall_confidence": sum(state["confidence_scores"].values()) / len(state["confidence_scores"]),
            "processing_summary": {
                "stages_completed": state["processing_stage"],
                "parallel_results": len(state["parallel_results"]),
                "errors_encountered": len(state["error_log"])
            }
        }
        
        return {
            **state,
            "trading_recommendations": trading_recommendations,
            "processing_stage": "complete",
            "real_time_updates": state["real_time_updates"] + [
                {"stage": "finalization", "timestamp": datetime.now().isoformat()}
            ]
        }
    
    async def _handle_errors(self, state: TradingState) -> TradingState:
        print(f"❌ Handling system errors: {len(state['error_log'])} errors")
        
        return {
            **state,
            "processing_stage": "error_handled",
            "trading_recommendations": {
                "error": True,
                "errors": state["error_log"],
                "partial_results": {
                    "market_analysis": state.get("market_analysis"),
                    "risk_assessment": state.get("risk_assessment")
                }
            }
        }
    
    # Subgraph node implementations
    async def _market_analysis_node(self, state: TradingState) -> TradingState:
        market_analysis = await self.market_agent.analyze_market_data(state["symbols"])
        return {**state, "market_analysis": market_analysis}
    
    async def _risk_analysis_node(self, state: TradingState) -> TradingState:
        risk_assessment = await self.risk_agent.assess_portfolio_risk(state["portfolio"], state["market_analysis"])
        return {**state, "risk_assessment": risk_assessment}
    
    async def _generate_signals_node(self, state: TradingState) -> TradingState:
        strategy_signals = await self.strategy_agent.generate_trading_signals(state["market_analysis"], state["risk_assessment"])
        return {**state, "strategy_signals": strategy_signals}
    
    async def _validate_strategies_node(self, state: TradingState) -> TradingState:
        # Add strategy validation logic here
        return state
    
    async def run_trading_system(self, symbols: List[str], portfolio: Dict[str, Any], trading_mode: str = "analysis") -> Dict[str, Any]:
        """Main entry point for the advanced trading system"""
        
        initial_state = {
            "symbols": symbols,
            "portfolio": portfolio,
            "trading_mode": trading_mode,
            "user_preferences": {},
            "market_analysis": {},
            "risk_assessment": {},
            "strategy_signals": {},
            "execution_plan": {},
            "processing_stage": "starting",
            "parallel_results": [],
            "error_log": [],
            "confidence_scores": {},
            "trading_recommendations": {},
            "performance_metrics": {},
            "real_time_updates": []
        }
        
        result = await self.main_workflow.ainvoke(initial_state)
        return result

# Test the advanced workflow
if __name__ == "__main__":
    async def test_advanced_workflow():
        workflow = AdvancedTradingWorkflow()
        
        print("🚀 Testing Advanced LangGraph Trading Workflow...")
        
        test_symbols = ["AAPL", "GOOGL", "TSLA", "NVDA"]
        test_portfolio = {
            "total_value": 100000,
            "positions": {
                "AAPL": 25000,
                "GOOGL": 30000,
                "cash": 45000
            }
        }
        
        result = await workflow.run_trading_system(test_symbols, test_portfolio)
        
        print("✅ Advanced workflow complete!")
        print(f"📊 Processing stage: {result['processing_stage']}")
        print(f"⚡ Parallel results: {len(result['parallel_results'])}")
        print(f"🎯 Trading opportunities: {len(result.get('trading_recommendations', {}).get('top_opportunities', []))}")
        print(f"🔥 Overall confidence: {result.get('trading_recommendations', {}).get('overall_confidence', 0):.1%}")
        
        return result
    
    # Run the test
    asyncio.run(test_advanced_workflow())