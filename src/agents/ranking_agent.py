"""
Ranking Agent for tournament-based strategy evaluation and evolution.
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

from .base_agent import BaseAgent
from ..utils.config import load_config

logger = logging.getLogger(__name__)

class RankingAgent(BaseAgent):
    """
    Agent for managing strategy tournaments and rankings.
    
    Attributes:
        name: Agent identifier
        config: Configuration dictionary
        tournament_history: History of tournaments
        current_rankings: Current strategy rankings
    """
    
    def __init__(
        self,
        name: str = "ranking_agent",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize ranking agent."""
        from .base_agent import AgentType
        super().__init__(
            name=name,
            agent_type=AgentType.RANKING,
            config=config,
            **kwargs
        )
        
        # Initialize rankings
        self.current_rankings = {}
        self.tournament_history = []
        
        # Load ranking parameters
        self.ranking_params = self.config.get('ranking_params', {
            'k_factor': 32,           # Elo K-factor
            'tournament_size': 10,     # Strategies per tournament
            'min_matches': 5,          # Minimum matches for ranking
            'volatility_weight': 0.3,  # Weight for volatility in scoring
            'consistency_weight': 0.3, # Weight for consistency in scoring
            'returns_weight': 0.4      # Weight for returns in scoring
        })
        
        # Initialize metrics
        self.metrics.update({
            'tournaments_run': 0,
            'matches_evaluated': 0,
            'ranked_strategies': 0,
            'evolution_cycles': 0
        })
        
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process tournament or ranking request.
        
        Args:
            data: Request data including:
                - strategies: List of strategies to evaluate
                - market_data: Market data for evaluation
                - tournament_type: Type of tournament
                
        Returns:
            Tournament results and rankings
        """
        try:
            # Validate request
            if not self._validate_request(data):
                raise ValueError("Invalid tournament request")
                
            # Run tournament
            tournament_results = await self._run_tournament(
                strategies=data['strategies'],
                market_data=data['market_data'],
                tournament_type=data.get('tournament_type', 'round_robin')
            )
            
            # Update rankings
            self._update_rankings(tournament_results)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(tournament_results)
            
            # Store results
            tournament_id = self._store_tournament(
                tournament_results,
                recommendations,
                data
            )
            
            # Update metrics
            self._update_tournament_metrics(tournament_results)
            
            return {
                'status': 'success',
                'tournament_id': tournament_id,
                'results': tournament_results,
                'rankings': self.current_rankings,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Tournament failed: {str(e)}")
            self._log_error(e)
            raise
            
    async def _run_tournament(
        self,
        strategies: List[Dict[str, Any]],
        market_data: pd.DataFrame,
        tournament_type: str
    ) -> Dict[str, Any]:
        """
        Run tournament between strategies.
        
        Args:
            strategies: List of strategies to evaluate
            market_data: Market data for evaluation
            tournament_type: Tournament format
            
        Returns:
            Tournament results
        """
        if tournament_type == 'round_robin':
            return await self._run_round_robin(strategies, market_data)
        elif tournament_type == 'elimination':
            return await self._run_elimination(strategies, market_data)
        elif tournament_type == 'swiss':
            return await self._run_swiss(strategies, market_data)
        else:
            raise ValueError(f"Unknown tournament type: {tournament_type}")
            
    async def _run_round_robin(
        self,
        strategies: List[Dict[str, Any]],
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run round-robin tournament."""
        results = {
            'matches': [],
            'scores': {},
            'rankings': {}
        }
        
        # Initialize scores
        for strategy in strategies:
            results['scores'][strategy['id']] = 0
            
        # Run all matches
        for i, strategy1 in enumerate(strategies):
            for strategy2 in strategies[i+1:]:
                match_result = await self._run_match(
                    strategy1,
                    strategy2,
                    market_data
                )
                results['matches'].append(match_result)
                
                # Update scores
                results['scores'][strategy1['id']] += match_result['score1']
                results['scores'][strategy2['id']] += match_result['score2']
                
        # Calculate final rankings
        results['rankings'] = self._calculate_rankings(results['scores'])
        
        return results
        
    async def _run_elimination(
        self,
        strategies: List[Dict[str, Any]],
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run elimination tournament."""
        results = {
            'rounds': [],
            'matches': [],
            'rankings': {}
        }
        
        remaining_strategies = strategies.copy()
        
        while len(remaining_strategies) > 1:
            round_results = []
            next_round = []
            
            # Pair strategies and run matches
            for i in range(0, len(remaining_strategies), 2):
                if i + 1 >= len(remaining_strategies):
                    next_round.append(remaining_strategies[i])
                    continue
                    
                match_result = await self._run_match(
                    remaining_strategies[i],
                    remaining_strategies[i + 1],
                    market_data
                )
                round_results.append(match_result)
                
                # Winner advances
                winner = (remaining_strategies[i] if match_result['score1'] > match_result['score2']
                         else remaining_strategies[i + 1])
                next_round.append(winner)
                
            results['rounds'].append(round_results)
            results['matches'].extend(round_results)
            remaining_strategies = next_round
            
        return results
        
    async def _run_swiss(
        self,
        strategies: List[Dict[str, Any]],
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run Swiss-system tournament."""
        results = {
            'rounds': [],
            'matches': [],
            'scores': {},
            'rankings': {}
        }
        
        # Initialize scores
        for strategy in strategies:
            results['scores'][strategy['id']] = 0
            
        num_rounds = int(np.log2(len(strategies)))
        
        for round_num in range(num_rounds):
            # Pair strategies based on current scores
            pairs = self._create_swiss_pairs(
                strategies,
                results['scores'],
                results['matches']
            )
            
            round_results = []
            for strategy1, strategy2 in pairs:
                match_result = await self._run_match(
                    strategy1,
                    strategy2,
                    market_data
                )
                round_results.append(match_result)
                
                # Update scores
                results['scores'][strategy1['id']] += match_result['score1']
                results['scores'][strategy2['id']] += match_result['score2']
                
            results['rounds'].append(round_results)
            results['matches'].extend(round_results)
            
        results['rankings'] = self._calculate_rankings(results['scores'])
        return results
        
    async def _run_match(
        self,
        strategy1: Dict[str, Any],
        strategy2: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run match between two strategies."""
        # Get strategy performance
        perf1 = await self._evaluate_strategy(strategy1, market_data)
        perf2 = await self._evaluate_strategy(strategy2, market_data)
        
        # Calculate scores
        score1, score2 = self._calculate_match_scores(perf1, perf2)
        
        return {
            'strategy1_id': strategy1['id'],
            'strategy2_id': strategy2['id'],
            'score1': score1,
            'score2': score2,
            'performance1': perf1,
            'performance2': perf2,
            'timestamp': datetime.now().isoformat()
        }
        
    def _calculate_rankings(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """Calculate rankings from scores."""
        # Sort by score
        ranked_strategies = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        rankings = {}
        for rank, (strategy_id, score) in enumerate(ranked_strategies, 1):
            rankings[strategy_id] = {
                'rank': rank,
                'score': score,
                'previous_rank': self.current_rankings.get(
                    strategy_id, {}
                ).get('rank', rank)
            }
            
        return rankings
        
    def _update_rankings(self, tournament_results: Dict[str, Any]) -> None:
        """Update global rankings."""
        for strategy_id, ranking in tournament_results['rankings'].items():
            if strategy_id not in self.current_rankings:
                self.current_rankings[strategy_id] = {
                    'rank': ranking['rank'],
                    'score': ranking['score'],
                    'matches': 1,
                    'history': []
                }
            else:
                current = self.current_rankings[strategy_id]
                current['rank'] = ranking['rank']
                current['score'] = (
                    current['score'] * current['matches'] + ranking['score']
                ) / (current['matches'] + 1)
                current['matches'] += 1
                current['history'].append({
                    'timestamp': datetime.now().isoformat(),
                    'rank': ranking['rank'],
                    'score': ranking['score']
                })
                
    def _generate_recommendations(
        self,
        tournament_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate tournament recommendations."""
        recommendations = []
        
        # Identify top performers
        top_performers = self._identify_top_performers(tournament_results)
        if top_performers:
            recommendations.append({
                'type': 'evolution',
                'action': 'evolve_top_performers',
                'strategies': top_performers
            })
            
        # Identify underperformers
        underperformers = self._identify_underperformers(tournament_results)
        if underperformers:
            recommendations.append({
                'type': 'replacement',
                'action': 'replace_underperformers',
                'strategies': underperformers
            })
            
        # Identify promising combinations
        combinations = self._identify_promising_combinations(tournament_results)
        if combinations:
            recommendations.append({
                'type': 'combination',
                'action': 'combine_strategies',
                'pairs': combinations
            })
            
        return recommendations
        
    async def _evaluate_strategy(
        self,
        strategy: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Evaluate strategy performance."""
        # Request backtest from backtesting agent
        backtest_request = {
            'strategy': strategy,
            'market_data': market_data,
            'parameters': {
                'initial_capital': 100000,
                'trading_costs': True
            }
        }
        
        backtest_response = await self.send_message({
            'type': 'backtest_request',
            'data': backtest_request,
            'timestamp': datetime.now().isoformat()
        })
        
        if backtest_response['status'] != 'success':
            raise ValueError(f"Backtest failed: {backtest_response.get('error')}")
            
        return backtest_response['data']['analysis']
        
    def _calculate_match_scores(
        self,
        perf1: Dict[str, float],
        perf2: Dict[str, float]
    ) -> Tuple[float, float]:
        """Calculate match scores from performance metrics."""
        # Calculate weighted score
        score1 = (
            self.ranking_params['returns_weight'] * perf1['performance']['total_return'] +
            self.ranking_params['volatility_weight'] * (1 / perf1['risk']['volatility']) +
            self.ranking_params['consistency_weight'] * perf1['trades']['win_rate']
        )
        
        score2 = (
            self.ranking_params['returns_weight'] * perf2['performance']['total_return'] +
            self.ranking_params['volatility_weight'] * (1 / perf2['risk']['volatility']) +
            self.ranking_params['consistency_weight'] * perf2['trades']['win_rate']
        )
        
        return score1, score2
        
    def _create_swiss_pairs(
        self,
        strategies: List[Dict[str, Any]],
        scores: Dict[str, float],
        previous_matches: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Create Swiss tournament pairings."""
        # Sort strategies by score
        sorted_strategies = sorted(
            strategies,
            key=lambda s: scores[s['id']],
            reverse=True
        )
        
        # Create pairs avoiding rematches
        pairs = []
        used = set()
        
        for strategy in sorted_strategies:
            if strategy['id'] in used:
                continue
                
            # Find best unmatch opponent
            for opponent in sorted_strategies:
                if (opponent['id'] not in used and
                    opponent['id'] != strategy['id'] and
                    not self._have_matched(strategy, opponent, previous_matches)):
                    pairs.append((strategy, opponent))
                    used.add(strategy['id'])
                    used.add(opponent['id'])
                    break
                    
        return pairs
        
    def _have_matched(
        self,
        strategy1: Dict[str, Any],
        strategy2: Dict[str, Any],
        matches: List[Dict[str, Any]]
    ) -> bool:
        """Check if two strategies have already matched."""
        for match in matches:
            if (
                (match['strategy1_id'] == strategy1['id'] and
                 match['strategy2_id'] == strategy2['id']) or
                (match['strategy1_id'] == strategy2['id'] and
                 match['strategy2_id'] == strategy1['id'])
            ):
                return True
        return False
        
    def _identify_top_performers(
        self,
        tournament_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify top performing strategies."""
        return [
            {'id': strategy_id, **ranking}
            for strategy_id, ranking in tournament_results['rankings'].items()
            if ranking['rank'] <= 3
        ]
        
    def _identify_underperformers(
        self,
        tournament_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify underperforming strategies."""
        return [
            {'id': strategy_id, **ranking}
            for strategy_id, ranking in tournament_results['rankings'].items()
            if ranking['rank'] > len(tournament_results['rankings']) * 0.8
        ]
        
    def _identify_promising_combinations(
        self,
        tournament_results: Dict[str, Any]
    ) -> List[Tuple[str, str]]:
        """Identify promising strategy combinations."""
        combinations = []
        rankings = tournament_results['rankings']
        
        # Find complementary top performers
        top_strategies = [
            strategy_id for strategy_id, ranking in rankings.items()
            if ranking['rank'] <= 5
        ]
        
        for i, strategy1_id in enumerate(top_strategies):
            for strategy2_id in top_strategies[i+1:]:
                if self._are_complementary(
                    strategy1_id,
                    strategy2_id,
                    tournament_results
                ):
                    combinations.append((strategy1_id, strategy2_id))
                    
        return combinations
        
    def _are_complementary(
        self,
        strategy1_id: str,
        strategy2_id: str,
        tournament_results: Dict[str, Any]
    ) -> bool:
        """Check if strategies are complementary."""
        # Implement complementary strategy detection
        return False