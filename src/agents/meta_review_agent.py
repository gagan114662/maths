"""
Meta-Review Agent for analyzing patterns and synthesizing insights across strategies.
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .base_agent import BaseAgent
from ..utils.config import load_config

logger = logging.getLogger(__name__)

class MetaReviewAgent(BaseAgent):
    """
    Agent for meta-analysis of strategies and system performance.
    
    Attributes:
        name: Agent identifier
        config: Configuration dictionary
        analysis_history: History of meta-analyses
    """
    
    def __init__(
        self,
        name: str = "meta_review_agent",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize meta-review agent."""
        from .base_agent import AgentType
        super().__init__(
            name=name,
            agent_type=AgentType.META_REVIEW,
            config=config,
            **kwargs
        )
        
        # Initialize analysis history
        self.analysis_history = []
        
        # Load analysis parameters
        self.analysis_params = self.config.get('analysis_params', {
            'lookback_days': 30,
            'min_strategies': 5,
            'cluster_count': 3,
            'correlation_threshold': 0.7,
            'performance_threshold': 0.8
        })
        
        # Initialize metrics
        self.metrics.update({
            'analyses_performed': 0,
            'patterns_identified': 0,
            'recommendations_made': 0,
            'insights_generated': 0
        })
        
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process meta-analysis request.
        
        Args:
            data: Request data including:
                - strategies: Strategy performance data
                - market_data: Market data
                - system_metrics: System performance metrics
                
        Returns:
            Meta-analysis results
        """
        try:
            # Validate request
            if not self._validate_request(data):
                raise ValueError("Invalid meta-analysis request")
                
            # Perform analysis
            analysis = await self._analyze_system(
                strategies=data['strategies'],
                market_data=data['market_data'],
                system_metrics=data['system_metrics']
            )
            
            # Generate insights
            insights = self._generate_insights(analysis)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(analysis, insights)
            
            # Store results
            analysis_id = self._store_analysis(
                analysis,
                insights,
                recommendations,
                data
            )
            
            # Update metrics
            self._update_analysis_metrics(analysis)
            
            return {
                'status': 'success',
                'analysis_id': analysis_id,
                'analysis': analysis,
                'insights': insights,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Meta-analysis failed: {str(e)}")
            self._log_error(e)
            raise
            
    async def _analyze_system(
        self,
        strategies: List[Dict[str, Any]],
        market_data: Dict[str, Any],
        system_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform system-wide analysis.
        
        Args:
            strategies: Strategy performance data
            market_data: Market data
            system_metrics: System performance metrics
            
        Returns:
            System analysis
        """
        # Analyze strategy patterns
        strategy_patterns = self._analyze_strategy_patterns(strategies)
        
        # Analyze market influence
        market_influence = self._analyze_market_influence(
            strategies,
            market_data
        )
        
        # Analyze system performance
        system_performance = self._analyze_system_performance(
            system_metrics
        )
        
        # Perform correlation analysis
        correlations = self._analyze_correlations(
            strategies,
            market_data
        )
        
        # Cluster analysis
        clusters = self._perform_clustering(strategies)
        
        return {
            'strategy_patterns': strategy_patterns,
            'market_influence': market_influence,
            'system_performance': system_performance,
            'correlations': correlations,
            'clusters': clusters,
            'timestamp': datetime.now().isoformat()
        }
        
    def _analyze_strategy_patterns(
        self,
        strategies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze patterns in strategy behavior.
        
        Args:
            strategies: Strategy performance data
            
        Returns:
            Pattern analysis results
        """
        performance_data = self._extract_performance_data(strategies)
        
        return {
            'performance_trends': self._analyze_performance_trends(performance_data),
            'behavior_patterns': self._analyze_behavior_patterns(strategies),
            'commonalities': self._find_strategy_commonalities(strategies),
            'divergences': self._find_strategy_divergences(strategies)
        }
        
    def _analyze_market_influence(
        self,
        strategies: List[Dict[str, Any]],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze market influence on strategies.
        
        Args:
            strategies: Strategy performance data
            market_data: Market data
            
        Returns:
            Market influence analysis
        """
        return {
            'regime_impact': self._analyze_regime_impact(strategies, market_data),
            'market_sensitivity': self._analyze_market_sensitivity(strategies, market_data),
            'correlation_structure': self._analyze_correlation_structure(strategies, market_data),
            'adaptability': self._analyze_strategy_adaptability(strategies, market_data)
        }
        
    def _analyze_system_performance(
        self,
        system_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze overall system performance.
        
        Args:
            system_metrics: System performance metrics
            
        Returns:
            System performance analysis
        """
        return {
            'efficiency': self._analyze_system_efficiency(system_metrics),
            'reliability': self._analyze_system_reliability(system_metrics),
            'scalability': self._analyze_system_scalability(system_metrics),
            'robustness': self._analyze_system_robustness(system_metrics)
        }
        
    def _generate_insights(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate insights from analysis.
        
        Args:
            analysis: Analysis results
            
        Returns:
            List of insights
        """
        insights = []
        
        # Strategy insights
        strategy_insights = self._generate_strategy_insights(
            analysis['strategy_patterns']
        )
        insights.extend(strategy_insights)
        
        # Market insights
        market_insights = self._generate_market_insights(
            analysis['market_influence']
        )
        insights.extend(market_insights)
        
        # System insights
        system_insights = self._generate_system_insights(
            analysis['system_performance']
        )
        insights.extend(system_insights)
        
        # Risk insights
        risk_insights = self._generate_risk_insights(analysis)
        insights.extend(risk_insights)
        
        return insights
        
    def _generate_recommendations(
        self,
        analysis: Dict[str, Any],
        insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on analysis and insights.
        
        Args:
            analysis: Analysis results
            insights: Generated insights
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Strategy recommendations
        strategy_recommendations = self._generate_strategy_recommendations(
            analysis['strategy_patterns'],
            insights
        )
        recommendations.extend(strategy_recommendations)
        
        # System recommendations
        system_recommendations = self._generate_system_recommendations(
            analysis['system_performance'],
            insights
        )
        recommendations.extend(system_recommendations)
        
        # Risk management recommendations
        risk_recommendations = self._generate_risk_recommendations(
            analysis,
            insights
        )
        recommendations.extend(risk_recommendations)
        
        return recommendations
        
    def _perform_clustering(
        self,
        strategies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform clustering analysis on strategies.
        
        Args:
            strategies: Strategy data
            
        Returns:
            Clustering results
        """
        # Extract features for clustering
        features = self._extract_clustering_features(strategies)
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Perform clustering
        kmeans = KMeans(
            n_clusters=self.analysis_params['cluster_count'],
            random_state=42
        )
        clusters = kmeans.fit_predict(scaled_features)
        
        # Analyze clusters
        cluster_analysis = self._analyze_clusters(
            strategies,
            clusters,
            kmeans.cluster_centers_
        )
        
        return {
            'cluster_assignments': clusters.tolist(),
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'cluster_analysis': cluster_analysis
        }
        
    def _extract_clustering_features(
        self,
        strategies: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Extract features for clustering."""
        features = []
        for strategy in strategies:
            perf = strategy['performance']
            features.append([
                perf['total_return'],
                perf['sharpe_ratio'],
                perf['max_drawdown'],
                perf['volatility'],
                strategy['trades']['win_rate']
            ])
        return np.array(features)
        
    def _analyze_clusters(
        self,
        strategies: List[Dict[str, Any]],
        clusters: np.ndarray,
        centers: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Analyze cluster characteristics."""
        cluster_analysis = []
        
        for i in range(len(centers)):
            cluster_strategies = [
                s for s, c in zip(strategies, clusters) if c == i
            ]
            
            cluster_analysis.append({
                'size': len(cluster_strategies),
                'average_return': np.mean([
                    s['performance']['total_return']
                    for s in cluster_strategies
                ]),
                'average_sharpe': np.mean([
                    s['performance']['sharpe_ratio']
                    for s in cluster_strategies
                ]),
                'strategy_types': self._count_strategy_types(cluster_strategies)
            })
            
        return cluster_analysis
        
    def _store_analysis(
        self,
        analysis: Dict[str, Any],
        insights: List[Dict[str, Any]],
        recommendations: List[Dict[str, Any]],
        request_data: Dict[str, Any]
    ) -> int:
        """Store analysis results."""
        analysis_data = {
            'analysis': analysis,
            'insights': insights,
            'recommendations': recommendations,
            'request_data': request_data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in memory manager
        analysis_id = self.memory.store(
            'meta_analysis',
            content=analysis_data,
            metadata={
                'strategies_analyzed': len(request_data['strategies']),
                'patterns_found': len(analysis['strategy_patterns']),
                'insights_generated': len(insights),
                'recommendations_made': len(recommendations)
            }
        )
        
        # Add to history
        self.analysis_history.append({
            'analysis_id': analysis_id,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'patterns': len(analysis['strategy_patterns']),
                'insights': len(insights),
                'recommendations': len(recommendations)
            }
        })
        
        return analysis_id
        
    def _update_analysis_metrics(self, analysis: Dict[str, Any]) -> None:
        """Update agent metrics."""
        metrics = self.state['metrics']
        metrics['analyses_performed'] += 1
        metrics['patterns_identified'] += len(analysis['strategy_patterns'])
        metrics['insights_generated'] += len(analysis.get('insights', []))
        metrics['recommendations_made'] += len(analysis.get('recommendations', []))