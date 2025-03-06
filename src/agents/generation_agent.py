"""
Generation agent for creating new trading strategies using scientific methodology.
"""
import logging
import json
import asyncio
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import traceback

from .base_agent import BaseAgent, AgentType
from ..core.memory_manager import MemoryType, MemoryImportance

logger = logging.getLogger(__name__)

class HypothesisState:
    """Enumeration of hypothesis states in scientific method."""
    FORMULATED = "formulated"
    TESTING = "testing"
    VALIDATED = "validated"
    REJECTED = "rejected"
    REFINED = "refined"

class Hypothesis:
    """Scientific hypothesis for trading strategy development."""
    def __init__(
        self,
        statement: str,
        rationale: str,
        predictions: List[str],
        variables: Dict[str, Any],
        null_hypothesis: str,
        confidence_threshold: float = 0.95
    ):
        self.id = str(uuid.uuid4())
        self.statement = statement
        self.rationale = rationale
        self.predictions = predictions
        self.variables = variables
        self.null_hypothesis = null_hypothesis
        self.confidence_threshold = confidence_threshold
        self.state = HypothesisState.FORMULATED
        self.test_results = []
        self.confidence_level = 0.0
        self.p_value = None
        self.created_at = datetime.now()
        self.modified_at = datetime.now()
        self.rejected_reason = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert hypothesis to dictionary."""
        return {
            "id": self.id,
            "statement": self.statement,
            "rationale": self.rationale,
            "predictions": self.predictions,
            "variables": self.variables,
            "null_hypothesis": self.null_hypothesis,
            "confidence_threshold": self.confidence_threshold,
            "state": self.state,
            "test_results": self.test_results,
            "confidence_level": self.confidence_level,
            "p_value": self.p_value,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "rejected_reason": self.rejected_reason
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Hypothesis':
        """Create hypothesis from dictionary."""
        hypothesis = cls(
            statement=data["statement"],
            rationale=data["rationale"],
            predictions=data["predictions"],
            variables=data["variables"],
            null_hypothesis=data["null_hypothesis"],
            confidence_threshold=data["confidence_threshold"]
        )
        hypothesis.id = data["id"]
        hypothesis.state = data["state"]
        hypothesis.test_results = data["test_results"]
        hypothesis.confidence_level = data["confidence_level"]
        hypothesis.p_value = data["p_value"]
        hypothesis.created_at = datetime.fromisoformat(data["created_at"])
        hypothesis.modified_at = datetime.fromisoformat(data["modified_at"])
        hypothesis.rejected_reason = data["rejected_reason"]
        return hypothesis

class Experiment:
    """Scientific experiment for testing trading strategy hypotheses."""
    def __init__(
        self,
        hypothesis_id: str,
        design: Dict[str, Any],
        control_variables: Dict[str, Any],
        test_variables: Dict[str, Any],
        data_requirements: Dict[str, Any],
        evaluation_metrics: List[str],
        significance_level: float = 0.05
    ):
        self.id = str(uuid.uuid4())
        self.hypothesis_id = hypothesis_id
        self.design = design
        self.control_variables = control_variables
        self.test_variables = test_variables
        self.data_requirements = data_requirements
        self.evaluation_metrics = evaluation_metrics
        self.significance_level = significance_level
        self.status = "created"
        self.results = None
        self.start_time = None
        self.end_time = None
        self.conclusion = None
        self.reproducibility_seed = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary."""
        return {
            "id": self.id,
            "hypothesis_id": self.hypothesis_id,
            "design": self.design,
            "control_variables": self.control_variables,
            "test_variables": self.test_variables,
            "data_requirements": self.data_requirements,
            "evaluation_metrics": self.evaluation_metrics,
            "significance_level": self.significance_level,
            "status": self.status,
            "results": self.results,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "conclusion": self.conclusion,
            "reproducibility_seed": self.reproducibility_seed
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experiment':
        """Create experiment from dictionary."""
        experiment = cls(
            hypothesis_id=data["hypothesis_id"],
            design=data["design"],
            control_variables=data["control_variables"],
            test_variables=data["test_variables"],
            data_requirements=data["data_requirements"],
            evaluation_metrics=data["evaluation_metrics"],
            significance_level=data["significance_level"]
        )
        experiment.id = data["id"]
        experiment.status = data["status"]
        experiment.results = data["results"]
        experiment.start_time = datetime.fromisoformat(data["start_time"]) if data["start_time"] else None
        experiment.end_time = datetime.fromisoformat(data["end_time"]) if data["end_time"] else None
        experiment.conclusion = data["conclusion"]
        experiment.reproducibility_seed = data["reproducibility_seed"]
        return experiment

class ScientificMethod:
    """Implementation of scientific method for strategy development."""
    def __init__(self):
        self.hypotheses = {}  # id -> Hypothesis
        self.experiments = {}  # id -> Experiment
        self.observations = []
        self.literature_review = {}
        self.journal = []
        
    def add_hypothesis(self, hypothesis: Hypothesis) -> str:
        """Add hypothesis and return its ID."""
        self.hypotheses[hypothesis.id] = hypothesis
        self.journal.append({
            "timestamp": datetime.now().isoformat(),
            "action": "hypothesis_created",
            "hypothesis_id": hypothesis.id,
            "statement": hypothesis.statement
        })
        return hypothesis.id
        
    def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Get hypothesis by ID."""
        return self.hypotheses.get(hypothesis_id)
        
    def create_experiment(self, experiment: Experiment) -> str:
        """Add experiment and return its ID."""
        self.experiments[experiment.id] = experiment
        self.journal.append({
            "timestamp": datetime.now().isoformat(),
            "action": "experiment_created",
            "experiment_id": experiment.id,
            "hypothesis_id": experiment.hypothesis_id
        })
        return experiment.id
        
    def update_experiment_results(
        self, 
        experiment_id: str, 
        results: Dict[str, Any],
        conclusion: str
    ) -> bool:
        """Update experiment results and return success status."""
        if experiment_id not in self.experiments:
            return False
            
        experiment = self.experiments[experiment_id]
        experiment.results = results
        experiment.end_time = datetime.now()
        experiment.conclusion = conclusion
        experiment.status = "completed"
        
        # Update the associated hypothesis
        hypothesis = self.hypotheses.get(experiment.hypothesis_id)
        if hypothesis:
            hypothesis.test_results.append({
                "experiment_id": experiment_id,
                "results": results,
                "conclusion": conclusion,
                "timestamp": datetime.now().isoformat()
            })
            hypothesis.modified_at = datetime.now()
            
        self.journal.append({
            "timestamp": datetime.now().isoformat(),
            "action": "experiment_completed",
            "experiment_id": experiment_id,
            "hypothesis_id": experiment.hypothesis_id,
            "conclusion": conclusion
        })
        
        return True
        
    def update_hypothesis_state(
        self,
        hypothesis_id: str,
        new_state: str,
        confidence_level: float = None,
        p_value: float = None,
        rejected_reason: str = None
    ) -> bool:
        """Update hypothesis state and return success status."""
        if hypothesis_id not in self.hypotheses:
            return False
            
        hypothesis = self.hypotheses[hypothesis_id]
        hypothesis.state = new_state
        hypothesis.modified_at = datetime.now()
        
        if confidence_level is not None:
            hypothesis.confidence_level = confidence_level
            
        if p_value is not None:
            hypothesis.p_value = p_value
            
        if rejected_reason is not None:
            hypothesis.rejected_reason = rejected_reason
            
        self.journal.append({
            "timestamp": datetime.now().isoformat(),
            "action": "hypothesis_state_updated",
            "hypothesis_id": hypothesis_id,
            "new_state": new_state,
            "confidence_level": confidence_level,
            "p_value": p_value,
            "rejected_reason": rejected_reason
        })
        
        return True
        
    def refine_hypothesis(
        self,
        original_id: str,
        refinements: Dict[str, Any]
    ) -> Optional[str]:
        """Refine hypothesis and return new hypothesis ID."""
        if original_id not in self.hypotheses:
            return None
            
        original = self.hypotheses[original_id]
        
        # Create new hypothesis with refinements
        refined = Hypothesis(
            statement=refinements.get("statement", original.statement),
            rationale=refinements.get("rationale", original.rationale),
            predictions=refinements.get("predictions", original.predictions.copy()),
            variables=refinements.get("variables", original.variables.copy()),
            null_hypothesis=refinements.get("null_hypothesis", original.null_hypothesis),
            confidence_threshold=refinements.get("confidence_threshold", original.confidence_threshold)
        )
        
        # Add reference to original
        refined.variables["derived_from"] = original_id
        
        # Add to collection
        self.hypotheses[refined.id] = refined
        
        # Update original hypothesis state
        self.update_hypothesis_state(original_id, HypothesisState.REFINED)
        
        self.journal.append({
            "timestamp": datetime.now().isoformat(),
            "action": "hypothesis_refined",
            "original_id": original_id,
            "refined_id": refined.id,
            "refinements": refinements
        })
        
        return refined.id
        
    def get_hypothesis_lineage(self, hypothesis_id: str) -> List[str]:
        """Get the lineage of hypotheses (refinement history)."""
        if hypothesis_id not in self.hypotheses:
            return []
            
        lineage = [hypothesis_id]
        current = self.hypotheses[hypothesis_id]
        
        # Check for parent hypotheses (derived_from)
        while "derived_from" in current.variables:
            parent_id = current.variables["derived_from"]
            if parent_id in self.hypotheses:
                lineage.insert(0, parent_id)
                current = self.hypotheses[parent_id]
            else:
                break
                
        return lineage
        
    def get_scientific_report(self) -> Dict[str, Any]:
        """Generate scientific report of all hypotheses and experiments."""
        return {
            "hypotheses_count": len(self.hypotheses),
            "experiments_count": len(self.experiments),
            "validated_hypotheses": [h.to_dict() for h in self.hypotheses.values() if h.state == HypothesisState.VALIDATED],
            "rejected_hypotheses": [h.to_dict() for h in self.hypotheses.values() if h.state == HypothesisState.REJECTED],
            "testing_hypotheses": [h.to_dict() for h in self.hypotheses.values() if h.state == HypothesisState.TESTING],
            "recent_experiments": [e.to_dict() for e in list(self.experiments.values())[-10:]],
            "scientific_journal": self.journal[-50:]  # Last 50 entries
        }

class GenerationAgent(BaseAgent):
    """
    Agent for generating novel trading strategies using scientific methodology.
    
    This agent is responsible for formulating testable trading hypotheses,
    designing experiments to validate them, and developing scientifically
    sound trading strategies based on experimental evidence.
    """
    
    def __init__(self, name: str, **kwargs):
        """Initialize Generation Agent."""
        super().__init__(name, AgentType.GENERATION, **kwargs)
        
        # Update system prompt for scientific strategy generation
        self.system_prompt = self._get_generation_prompt()
        
        # Initialize scientific method framework
        self.scientific_method = ScientificMethod()
        
        # Initialize strategy idea cache
        self.strategy_ideas = []
        
    def _get_generation_prompt(self) -> str:
        """Get specialized system prompt for scientific strategy generation."""
        prompt = f"""You are a Strategy Generation Agent implementing the scientific method in an AI trading system. 
Your role is to formulate testable trading hypotheses, design experiments to validate them, and develop 
scientifically sound trading strategies based on experimental evidence.

Your scientific approach:
1. Observe market phenomena and identify patterns
2. Formulate precise, testable hypotheses about market behavior
3. Design controlled experiments to test hypotheses
4. Analyze results with statistical rigor
5. Accept, reject, or refine hypotheses based on evidence
6. Develop trading strategies from validated hypotheses

SCIENTIFIC REQUIREMENTS:
- Formulate hypotheses with clear null and alternative statements
- Specify precise, measurable predictions
- Design experiments with proper controls and variable isolation
- Apply appropriate statistical tests with significance thresholds
- Consider sample size, power, and effect size in analysis
- Ensure reproducibility by documenting all parameters and methods
- Actively seek to falsify hypotheses, not just confirm them
- Record all experimental steps and results for peer review

When generating strategies:
- Focus on realistic, implementable strategies with scientific foundation
- Include specific indicators, parameters, and thresholds
- Consider transaction costs and market impact
- Provide evidence-based rationale with statistical support
- Design strategies with clear risk management rules

Your output should follow scientific paper structure with methods, results, discussion, and limitations sections.
"""
        return prompt
    
    async def initialize(self) -> bool:
        """Perform additional initialization steps."""
        # Call parent initialization
        if not await super().initialize():
            return False
            
        # Load any previously generated strategies and hypotheses from memory
        try:
            # Load strategies
            previous_strategies = self.memory_manager.retrieve(
                memory_type=MemoryType.STRATEGY,
                limit=10,
                tags=["generation", "idea"]
            )
            
            # Add to local cache
            for strategy in previous_strategies:
                self.strategy_ideas.append(strategy["content"])
                
            # Load hypotheses
            hypothesis_records = self.memory_manager.retrieve(
                memory_type=MemoryType.STRATEGY,
                limit=50,
                tags=["hypothesis"]
            )
            
            # Restore scientific method state
            for record in hypothesis_records:
                try:
                    hypothesis_data = record["content"]
                    hypothesis = Hypothesis.from_dict(hypothesis_data)
                    self.scientific_method.hypotheses[hypothesis.id] = hypothesis
                except Exception as e:
                    logger.warning(f"Error loading hypothesis: {str(e)}")
                    
            # Load experiments
            experiment_records = self.memory_manager.retrieve(
                memory_type=MemoryType.STRATEGY,
                limit=100,
                tags=["experiment"]
            )
            
            for record in experiment_records:
                try:
                    experiment_data = record["content"]
                    experiment = Experiment.from_dict(experiment_data)
                    self.scientific_method.experiments[experiment.id] = experiment
                except Exception as e:
                    logger.warning(f"Error loading experiment: {str(e)}")
            
            logger.info(f"Loaded {len(previous_strategies)} strategies, {len(hypothesis_records)} hypotheses, and {len(experiment_records)} experiments")
            return True
            
        except Exception as e:
            logger.error(f"Error loading previous data: {str(e)}")
            self._log_error(e)
            return False
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process generation requests.
        
        Args:
            data: Input data dictionary with request parameters
            
        Returns:
            Processing results with generated strategies or hypotheses
        """
        # Update metrics
        self._update_metrics({
            "requests_received": self.metrics.get("requests_received", 0) + 1
        })
        
        # Extract request details
        request_type = data.get("request_type", "generate")
        
        if request_type == "generate_strategy":
            return await self._generate_strategy(data)
        elif request_type == "get_strategies":
            return await self._get_strategy_ideas(data)
        elif request_type == "refine_strategy":
            return await self._refine_strategy(data)
        elif request_type == "formulate_hypothesis":
            return await self._formulate_hypothesis(data)
        elif request_type == "design_experiment":
            return await self._design_experiment(data)
        elif request_type == "analyze_results":
            return await self._analyze_results(data)
        elif request_type == "get_scientific_report":
            return await self._get_scientific_report(data)
        else:
            raise ValueError(f"Unknown request type: {request_type}")
    
    async def _generate_strategy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate new trading strategy ideas."""
        try:
            # Extract parameters
            market_data = data.get("market_data", {})
            constraints = data.get("constraints", {})
            count = data.get("count", 1)
            scientific_approach = data.get("scientific_approach", True)
            
            # Prepare prompt for strategy generation
            if scientific_approach:
                user_prompt = self._create_scientific_generation_prompt(market_data, constraints, count)
            else:
                user_prompt = self._create_generation_prompt(market_data, constraints, count)
            
            # Get response from LLM
            response = await self.chat_with_llm(user_prompt)
            
            # Parse strategies from response
            strategies = self._parse_strategies(response)
            
            # Store strategies in memory
            for strategy in strategies:
                # Add scientific metadata if using scientific approach
                if scientific_approach:
                    strategy["scientific_metadata"] = {
                        "hypothesis_formulated": True,
                        "tested": False,
                        "validated": False,
                        "confidence_level": None,
                        "p_value": None,
                        "effect_size": None
                    }
                
                self.memory_manager.store(
                    memory_type=MemoryType.STRATEGY,
                    content=strategy,
                    importance=MemoryImportance.MEDIUM,
                    metadata={
                        "generated_at": datetime.now().isoformat(),
                        "constraints": constraints,
                        "scientific_approach": scientific_approach
                    },
                    tags=["generation", "idea", "scientific" if scientific_approach else "heuristic"]
                )
                
            # Add to local cache
            self.strategy_ideas.extend(strategies)
            
            # Update metrics
            self._update_metrics({
                "strategies_generated": self.metrics.get("strategies_generated", 0) + len(strategies),
                "scientific_strategies": self.metrics.get("scientific_strategies", 0) + (len(strategies) if scientific_approach else 0)
            })
            
            return {
                "success": True,
                "strategies": strategies,
                "count": len(strategies),
                "scientific_approach": scientific_approach
            }
            
        except Exception as e:
            logger.error(f"Error generating strategies: {str(e)}")
            logger.debug(traceback.format_exc())
            self._log_error(e)
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_generation_prompt(
        self, 
        market_data: Dict[str, Any], 
        constraints: Dict[str, Any],
        count: int
    ) -> str:
        """Create prompt for standard strategy generation."""
        prompt = f"""Please generate {count} trading strategy ideas based on the following information and constraints.

MARKET DATA:
"""
        
        # Add market data if available
        if market_data:
            prompt += json.dumps(market_data, indent=2) + "\n\n"
        else:
            prompt += "No specific market data provided. Generate general strategies.\n\n"
            
        # Add constraints
        prompt += "CONSTRAINTS:\n"
        if constraints:
            for key, value in constraints.items():
                prompt += f"- {key}: {value}\n"
        else:
            prompt += "No specific constraints provided.\n"
            
        # Add output format instructions
        prompt += """
For each strategy, please provide:
1. Strategy Name: A descriptive name
2. Strategy Type: (Trend-following, Mean-reversion, Volatility-based, etc.)
3. Time Horizon: (Intraday, Daily, Weekly, Monthly)
4. Description: A detailed description of the strategy
5. Entry Rules: Specific conditions for entering positions
6. Exit Rules: Specific conditions for exiting positions
7. Position Sizing: How position size is determined
8. Risk Management: Specific risk management rules
9. Required Indicators: Technical indicators and parameters
10. Rationale: Why this strategy might work

Format each strategy as a JSON object with these fields.
"""
        
        return prompt
        
    def _create_scientific_generation_prompt(
        self, 
        market_data: Dict[str, Any], 
        constraints: Dict[str, Any],
        count: int
    ) -> str:
        """Create prompt for scientific strategy generation."""
        prompt = f"""Please generate {count} scientifically sound trading strategy ideas based on the following information and constraints.

MARKET DATA:
"""
        
        # Add market data if available
        if market_data:
            prompt += json.dumps(market_data, indent=2) + "\n\n"
        else:
            prompt += "No specific market data provided. Generate general strategies.\n\n"
            
        # Add constraints
        prompt += "CONSTRAINTS:\n"
        if constraints:
            for key, value in constraints.items():
                prompt += f"- {key}: {value}\n"
        else:
            prompt += "No specific constraints provided.\n"
            
        # Add scientific approach instructions
        prompt += """
SCIENTIFIC APPROACH:
Your strategy generation must follow the scientific method:
1. Base strategies on clearly formulated market hypotheses
2. Ensure hypotheses are testable and falsifiable
3. Define precise statistical measures and thresholds
4. Identify control variables and isolate test variables
5. Specify how to measure statistical significance
6. Include effect size considerations
7. Address potential biases and confounding factors

For each strategy, provide:
1. Strategy Name: A descriptive name
2. Strategy Type: (Trend-following, Mean-reversion, Volatility-based, etc.)
3. Time Horizon: (Intraday, Daily, Weekly, Monthly)
4. Market Hypothesis: The core scientific hypothesis behind the strategy
5. Null Hypothesis: The corresponding null hypothesis
6. Predictions: Specific, measurable predictions if the hypothesis is true
7. Test Methodology: How to validate the hypothesis empirically
8. Statistical Framework: Measures and significance thresholds
9. Entry Rules: Specific conditions for entering positions
10. Exit Rules: Specific conditions for exiting positions
11. Position Sizing: How position size is determined
12. Risk Management: Specific risk management rules
13. Required Indicators: Technical indicators and parameters
14. Potential Confounding Factors: What might invalidate results
15. Expected Effect Size: Anticipated magnitude of the edge
16. Alternative Explanations: Other possible interpretations of findings

Format each strategy as a JSON object with these fields.
"""
        
        return prompt
    
    def _parse_strategies(self, response: str) -> List[Dict[str, Any]]:
        """Parse strategies from LLM response."""
        strategies = []
        
        try:
            # Check if response contains JSON
            if "```json" in response:
                # Extract JSON blocks
                json_blocks = []
                lines = response.split("\n")
                in_json_block = False
                current_block = []
                
                for line in lines:
                    if line.strip() == "```json" or line.strip() == "```" and in_json_block:
                        in_json_block = not in_json_block
                        if not in_json_block and current_block:
                            json_blocks.append("\n".join(current_block))
                            current_block = []
                    elif in_json_block:
                        current_block.append(line)
                
                # Parse each JSON block
                for block in json_blocks:
                    try:
                        strategy = json.loads(block)
                        strategies.append(strategy)
                    except:
                        pass
                        
            # If no JSON blocks found or parsing failed, try to parse whole response
            if not strategies:
                try:
                    strategies = json.loads(response)
                    # Check if it's a dict (single strategy) or list (multiple strategies)
                    if isinstance(strategies, dict):
                        strategies = [strategies]
                except:
                    pass
            
            # If still no strategies, extract structured data manually
            if not strategies:
                # Implement manual parsing logic if needed
                pass
                
            return strategies
                
        except Exception as e:
            logger.error(f"Error parsing strategies: {str(e)}")
            return []
    
    async def _get_strategy_ideas(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get previously generated strategy ideas."""
        try:
            # Extract parameters
            limit = data.get("limit", 10)
            offset = data.get("offset", 0)
            filters = data.get("filters", {})
            
            # Apply filters if needed
            filtered_strategies = self.strategy_ideas
            
            # Filter by type if specified
            if "strategy_type" in filters:
                strategy_type = filters["strategy_type"]
                filtered_strategies = [
                    s for s in filtered_strategies 
                    if s.get("Strategy Type", "").lower() == strategy_type.lower()
                ]
                
            # Filter by time horizon if specified
            if "time_horizon" in filters:
                time_horizon = filters["time_horizon"]
                filtered_strategies = [
                    s for s in filtered_strategies 
                    if s.get("Time Horizon", "").lower() == time_horizon.lower()
                ]
                
            # Filter by scientific approach if specified
            if "scientific" in filters:
                scientific = filters["scientific"]
                filtered_strategies = [
                    s for s in filtered_strategies 
                    if bool(s.get("scientific_metadata", False)) == scientific
                ]
            
            # Apply pagination
            paginated_strategies = filtered_strategies[offset:offset+limit]
            
            return {
                "success": True,
                "strategies": paginated_strategies,
                "total": len(filtered_strategies),
                "limit": limit,
                "offset": offset
            }
            
        except Exception as e:
            logger.error(f"Error retrieving strategy ideas: {str(e)}")
            self._log_error(e)
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _refine_strategy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Refine an existing strategy based on feedback."""
        try:
            # Extract parameters
            strategy = data.get("strategy", {})
            feedback = data.get("feedback", "")
            scientific_approach = data.get("scientific_approach", True)
            
            if not strategy:
                raise ValueError("No strategy provided for refinement")
                
            if not feedback:
                raise ValueError("No feedback provided for refinement")
            
            # Prepare prompt for strategy refinement
            if scientific_approach:
                user_prompt = self._create_scientific_refinement_prompt(strategy, feedback)
            else:
                user_prompt = self._create_standard_refinement_prompt(strategy, feedback)
            
            # Get response from LLM
            response = await self.chat_with_llm(user_prompt)
            
            # Parse refined strategy from response
            refined_strategies = self._parse_strategies(response)
            
            if not refined_strategies:
                raise ValueError("Failed to parse refined strategy from response")
                
            refined_strategy = refined_strategies[0]
            
            # Store refined strategy in memory
            self.memory_manager.store(
                memory_type=MemoryType.STRATEGY,
                content=refined_strategy,
                importance=MemoryImportance.MEDIUM,
                metadata={
                    "refined_at": datetime.now().isoformat(),
                    "original_strategy": strategy,
                    "feedback": feedback,
                    "scientific_approach": scientific_approach
                },
                tags=["generation", "refinement", "scientific" if scientific_approach else "heuristic"]
            )
            
            # Update metrics
            self._update_metrics({
                "strategies_refined": self.metrics.get("strategies_refined", 0) + 1,
                "scientific_refinements": self.metrics.get("scientific_refinements", 0) + (1 if scientific_approach else 0)
            })
            
            return {
                "success": True,
                "original_strategy": strategy,
                "refined_strategy": refined_strategy,
                "feedback": feedback,
                "scientific_approach": scientific_approach
            }
            
        except Exception as e:
            logger.error(f"Error refining strategy: {str(e)}")
            self._log_error(e)
            
            return {
                "success": False,
                "error": str(e)
            }
            
    def _create_standard_refinement_prompt(
        self,
        strategy: Dict[str, Any],
        feedback: str
    ) -> str:
        """Create prompt for standard strategy refinement."""
        return f"""Please refine the following trading strategy based on the provided feedback.

STRATEGY:
{json.dumps(strategy, indent=2)}

FEEDBACK:
{feedback}

Please provide the refined strategy in the same JSON format as the original, with all fields updated as needed.
"""
            
    def _create_scientific_refinement_prompt(
        self,
        strategy: Dict[str, Any],
        feedback: str
    ) -> str:
        """Create prompt for scientific strategy refinement."""
        return f"""Please refine the following trading strategy scientifically based on the provided feedback.

STRATEGY:
{json.dumps(strategy, indent=2)}

FEEDBACK:
{feedback}

SCIENTIFIC REFINEMENT APPROACH:
1. Treat the feedback as experimental evidence
2. Update the market hypothesis based on this new evidence
3. Adjust predictions to account for the new information
4. Refine the test methodology to address any methodological issues
5. Update statistical measures and thresholds if needed
6. Revise entry/exit rules based on empirical findings
7. Document which aspects of the original hypothesis were supported or refuted
8. Consider alternative explanations for the observed results
9. Propose new experiments to further validate the refined strategy

Please provide the refined strategy in the same JSON format as the original, updating ALL fields to reflect the scientific refinement process.
"""
    
    async def _formulate_hypothesis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Formulate a scientific hypothesis for a trading strategy."""
        try:
            # Extract parameters
            market_observation = data.get("market_observation", "")
            prior_knowledge = data.get("prior_knowledge", {})
            constraints = data.get("constraints", {})
            
            if not market_observation:
                raise ValueError("No market observation provided for hypothesis formulation")
                
            # Prepare prompt for hypothesis formulation
            user_prompt = f"""Please formulate a scientific hypothesis for a trading strategy based on the following market observation.

MARKET OBSERVATION:
{market_observation}

PRIOR KNOWLEDGE:
{json.dumps(prior_knowledge, indent=2) if prior_knowledge else "No specific prior knowledge provided."}

CONSTRAINTS:
{json.dumps(constraints, indent=2) if constraints else "No specific constraints provided."}

Please formulate a proper scientific hypothesis with the following components:
1. Hypothesis Statement: A clear, precise statement of the proposed relationship or effect
2. Null Hypothesis: The corresponding null hypothesis (what would be true if the effect doesn't exist)
3. Rationale: Why this hypothesis is reasonable based on market mechanics or behavior
4. Testable Predictions: Specific, measurable outcomes that would confirm the hypothesis
5. Variables: Independent, dependent, and control variables
6. Confidence Threshold: Required confidence level to accept the hypothesis (e.g., 95%)

Format your response as a JSON object with these fields.
"""
            
            # Get response from LLM
            response = await self.chat_with_llm(user_prompt)
            
            # Parse hypothesis from response
            hypothesis_data = None
            
            try:
                # Check if response contains JSON
                if "```json" in response:
                    # Extract JSON block
                    json_start = response.find("```json") + 7
                    json_end = response.find("```", json_start)
                    if json_end > json_start:
                        hypothesis_data = json.loads(response[json_start:json_end].strip())
                else:
                    # Try to parse whole response
                    hypothesis_data = json.loads(response)
            except:
                logger.warning("Failed to parse hypothesis as JSON, attempting manual extraction")
                # Handle unstructured response manually
                pass
                
            if not hypothesis_data:
                raise ValueError("Failed to parse hypothesis from response")
                
            # Create hypothesis object
            hypothesis = Hypothesis(
                statement=hypothesis_data.get("Hypothesis Statement", ""),
                rationale=hypothesis_data.get("Rationale", ""),
                predictions=hypothesis_data.get("Testable Predictions", []),
                variables=hypothesis_data.get("Variables", {}),
                null_hypothesis=hypothesis_data.get("Null Hypothesis", ""),
                confidence_threshold=hypothesis_data.get("Confidence Threshold", 0.95)
            )
            
            # Add to scientific method framework
            hypothesis_id = self.scientific_method.add_hypothesis(hypothesis)
            
            # Store in memory
            self.memory_manager.store(
                memory_type=MemoryType.STRATEGY,
                content=hypothesis.to_dict(),
                importance=MemoryImportance.HIGH,
                metadata={
                    "formulated_at": datetime.now().isoformat(),
                    "market_observation": market_observation,
                    "prior_knowledge": prior_knowledge,
                    "constraints": constraints
                },
                tags=["hypothesis", "scientific", "generation"]
            )
            
            # Update metrics
            self._update_metrics({
                "hypotheses_formulated": self.metrics.get("hypotheses_formulated", 0) + 1
            })
            
            return {
                "success": True,
                "hypothesis_id": hypothesis_id,
                "hypothesis": hypothesis.to_dict(),
                "market_observation": market_observation
            }
            
        except Exception as e:
            logger.error(f"Error formulating hypothesis: {str(e)}")
            self._log_error(e)
            
            return {
                "success": False,
                "error": str(e)
            }
            
    async def _design_experiment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Design an experiment to test a hypothesis."""
        try:
            # Extract parameters
            hypothesis_id = data.get("hypothesis_id", "")
            data_constraints = data.get("data_constraints", {})
            evaluation_metrics = data.get("evaluation_metrics", [])
            
            if not hypothesis_id:
                raise ValueError("No hypothesis ID provided for experiment design")
                
            # Get hypothesis
            hypothesis = self.scientific_method.get_hypothesis(hypothesis_id)
            if not hypothesis:
                raise ValueError(f"Hypothesis with ID {hypothesis_id} not found")
                
            # Prepare prompt for experiment design
            user_prompt = f"""Please design a scientific experiment to test the following trading strategy hypothesis.

HYPOTHESIS:
{json.dumps(hypothesis.to_dict(), indent=2)}

DATA CONSTRAINTS:
{json.dumps(data_constraints, indent=2) if data_constraints else "No specific data constraints provided."}

EVALUATION METRICS:
{json.dumps(evaluation_metrics, indent=2) if evaluation_metrics else "No specific evaluation metrics provided."}

Please design a proper scientific experiment with the following components:
1. Experiment Design: Overall approach and methodology
2. Control Variables: Variables to hold constant across all tests
3. Test Variables: Variables to manipulate to test the hypothesis
4. Data Requirements: Time periods, assets, frequency, etc.
5. Evaluation Metrics: How to measure success/failure
6. Significance Level: Required p-value for statistical significance (e.g., 0.05)
7. Reproducibility Seed: Random seed for reproducibility

The experiment should follow best scientific practices:
- Properly isolate variables to test causal relationships
- Include appropriate control comparisons
- Use sufficient sample sizes for statistical power
- Account for multiple hypothesis testing if applicable
- Document potential biases and how they're addressed

Format your response as a JSON object with these fields.
"""
            
            # Get response from LLM
            response = await self.chat_with_llm(user_prompt)
            
            # Parse experiment from response
            experiment_data = None
            
            try:
                # Check if response contains JSON
                if "```json" in response:
                    # Extract JSON block
                    json_start = response.find("```json") + 7
                    json_end = response.find("```", json_start)
                    if json_end > json_start:
                        experiment_data = json.loads(response[json_start:json_end].strip())
                else:
                    # Try to parse whole response
                    experiment_data = json.loads(response)
            except:
                logger.warning("Failed to parse experiment as JSON, attempting manual extraction")
                # Handle unstructured response manually
                pass
                
            if not experiment_data:
                raise ValueError("Failed to parse experiment from response")
                
            # Create experiment object
            experiment = Experiment(
                hypothesis_id=hypothesis_id,
                design=experiment_data.get("Experiment Design", {}),
                control_variables=experiment_data.get("Control Variables", {}),
                test_variables=experiment_data.get("Test Variables", {}),
                data_requirements=experiment_data.get("Data Requirements", {}),
                evaluation_metrics=experiment_data.get("Evaluation Metrics", []),
                significance_level=experiment_data.get("Significance Level", 0.05)
            )
            
            # Set reproducibility seed if provided
            if "Reproducibility Seed" in experiment_data:
                experiment.reproducibility_seed = experiment_data["Reproducibility Seed"]
                
            # Add to scientific method framework
            experiment_id = self.scientific_method.create_experiment(experiment)
            
            # Update hypothesis state
            self.scientific_method.update_hypothesis_state(
                hypothesis_id=hypothesis_id,
                new_state=HypothesisState.TESTING
            )
            
            # Store in memory
            self.memory_manager.store(
                memory_type=MemoryType.STRATEGY,
                content=experiment.to_dict(),
                importance=MemoryImportance.HIGH,
                metadata={
                    "designed_at": datetime.now().isoformat(),
                    "hypothesis_id": hypothesis_id,
                    "data_constraints": data_constraints,
                    "evaluation_metrics": evaluation_metrics
                },
                tags=["experiment", "scientific", "generation"]
            )
            
            # Update metrics
            self._update_metrics({
                "experiments_designed": self.metrics.get("experiments_designed", 0) + 1
            })
            
            return {
                "success": True,
                "experiment_id": experiment_id,
                "experiment": experiment.to_dict(),
                "hypothesis_id": hypothesis_id
            }
            
        except Exception as e:
            logger.error(f"Error designing experiment: {str(e)}")
            self._log_error(e)
            
            return {
                "success": False,
                "error": str(e)
            }
            
    async def _analyze_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze experiment results and update hypothesis status."""
        try:
            # Extract parameters
            experiment_id = data.get("experiment_id", "")
            results = data.get("results", {})
            
            if not experiment_id:
                raise ValueError("No experiment ID provided for results analysis")
                
            if not results:
                raise ValueError("No results provided for analysis")
                
            # Get experiment
            experiment = self.scientific_method.experiments.get(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment with ID {experiment_id} not found")
                
            # Get associated hypothesis
            hypothesis = self.scientific_method.get_hypothesis(experiment.hypothesis_id)
            if not hypothesis:
                raise ValueError(f"Hypothesis with ID {experiment.hypothesis_id} not found")
                
            # Prepare prompt for results analysis
            user_prompt = f"""Please analyze the following experiment results for a trading strategy hypothesis and provide a scientific conclusion.

HYPOTHESIS:
{json.dumps(hypothesis.to_dict(), indent=2)}

EXPERIMENT:
{json.dumps(experiment.to_dict(), indent=2)}

RESULTS:
{json.dumps(results, indent=2)}

Please provide a rigorous scientific analysis with the following components:
1. Statistical Analysis: Detailed analysis of the results with appropriate statistical tests
2. Confidence Level: Calculated confidence level based on results
3. P-Value: Calculated p-value for the hypothesis test
4. Effect Size: Measured effect size and its practical significance
5. Conclusion: Whether to accept or reject the hypothesis based on the results
6. Rationale: Justification for the conclusion
7. Limitations: Limitations of the experiment and analysis
8. Next Steps: Recommended next steps (refine hypothesis, additional experiments, etc.)

The analysis should follow best scientific practices:
- Apply appropriate statistical tests for the data type
- Consider statistical power and sample size
- Address potential biases and confounding factors
- Distinguish between statistical and practical significance
- Consider alternative explanations for the results

Format your response as a JSON object with these fields.
"""
            
            # Get response from LLM
            response = await self.chat_with_llm(user_prompt)
            
            # Parse analysis from response
            analysis_data = None
            
            try:
                # Check if response contains JSON
                if "```json" in response:
                    # Extract JSON block
                    json_start = response.find("```json") + 7
                    json_end = response.find("```", json_start)
                    if json_end > json_start:
                        analysis_data = json.loads(response[json_start:json_end].strip())
                else:
                    # Try to parse whole response
                    analysis_data = json.loads(response)
            except:
                logger.warning("Failed to parse analysis as JSON, attempting manual extraction")
                # Handle unstructured response manually
                pass
                
            if not analysis_data:
                raise ValueError("Failed to parse analysis from response")
                
            # Extract key information
            conclusion = analysis_data.get("Conclusion", "").lower()
            confidence_level = analysis_data.get("Confidence Level")
            p_value = analysis_data.get("P-Value")
            
            # Try to convert confidence level to float if it's a string with percentage
            if isinstance(confidence_level, str):
                try:
                    confidence_level = float(confidence_level.strip("%")) / 100
                except:
                    pass
                    
            # Try to convert p-value to float if it's a string
            if isinstance(p_value, str):
                try:
                    p_value = float(p_value)
                except:
                    pass
                    
            # Determine new hypothesis state
            if "accept" in conclusion or "confirm" in conclusion or "support" in conclusion:
                new_state = HypothesisState.VALIDATED
                rejected_reason = None
            else:
                new_state = HypothesisState.REJECTED
                rejected_reason = analysis_data.get("Rationale", "")
                
            # Update experiment with results and conclusion
            self.scientific_method.update_experiment_results(
                experiment_id=experiment_id,
                results=results,
                conclusion=analysis_data.get("Conclusion", "")
            )
            
            # Update hypothesis state
            self.scientific_method.update_hypothesis_state(
                hypothesis_id=experiment.hypothesis_id,
                new_state=new_state,
                confidence_level=confidence_level if isinstance(confidence_level, float) else None,
                p_value=p_value if isinstance(p_value, float) else None,
                rejected_reason=rejected_reason
            )
            
            # Store analysis in memory
            self.memory_manager.store(
                memory_type=MemoryType.STRATEGY,
                content={
                    "experiment_id": experiment_id,
                    "hypothesis_id": experiment.hypothesis_id,
                    "results": results,
                    "analysis": analysis_data,
                    "conclusion": analysis_data.get("Conclusion", ""),
                    "new_state": new_state
                },
                importance=MemoryImportance.HIGH,
                metadata={
                    "analyzed_at": datetime.now().isoformat(),
                    "confidence_level": confidence_level,
                    "p_value": p_value
                },
                tags=["analysis", "scientific", "experiment"]
            )
            
            # Update metrics
            self._update_metrics({
                "results_analyzed": self.metrics.get("results_analyzed", 0) + 1,
                "hypotheses_validated": self.metrics.get("hypotheses_validated", 0) + (1 if new_state == HypothesisState.VALIDATED else 0),
                "hypotheses_rejected": self.metrics.get("hypotheses_rejected", 0) + (1 if new_state == HypothesisState.REJECTED else 0)
            })
            
            # Check if next steps recommend refinement
            next_steps = analysis_data.get("Next Steps", "")
            refinement_recommended = "refine" in next_steps.lower()
            
            return {
                "success": True,
                "experiment_id": experiment_id,
                "hypothesis_id": experiment.hypothesis_id,
                "conclusion": analysis_data.get("Conclusion", ""),
                "confidence_level": confidence_level,
                "p_value": p_value,
                "new_state": new_state,
                "refinement_recommended": refinement_recommended,
                "analysis": analysis_data
            }
            
        except Exception as e:
            logger.error(f"Error analyzing results: {str(e)}")
            self._log_error(e)
            
            return {
                "success": False,
                "error": str(e)
            }
            
    async def _get_scientific_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get a scientific report on all hypotheses and experiments."""
        try:
            # Generate report from scientific method framework
            report = self.scientific_method.get_scientific_report()
            
            # Add agent metrics
            report["agent_metrics"] = self.metrics
            
            return {
                "success": True,
                "report": report,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating scientific report: {str(e)}")
            self._log_error(e)
            
            return {
                "success": False,
                "error": str(e)
            }