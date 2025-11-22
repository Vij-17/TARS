"""Agent controller module.

This file primarily contains the project's neural AgentController implementation.
Lightweight, prototype controllers live in `proto_agent.py` to avoid naming
collisions during tests and imports.
"""
"""Agent Controller - Main agentic loop"""

import torch
import torch.nn as nn
from .goal_understanding import GoalUnderstanding
from .planning import PlanningModule
from .action_executor import ActionExecutor
from .state_tracker import StateTracker
from .reflection import ReflectionModule


class AgentController(nn.Module):
    """
    Main agentic control loop: Plan → Act → Observe → Reflect → Repeat
    """
    
    def __init__(
        self,
        dim=512,
        max_iterations=20,
        vocab_size=50000
    ):
        super().__init__()
        
        self.dim = dim
        self.max_iterations = max_iterations
        self.vocab_size = vocab_size
        
        # Agentic components
        self.goal_understanding = GoalUnderstanding(dim)
        self.planning_module = PlanningModule(dim)
        self.action_executor = ActionExecutor(dim)
        self.state_tracker = StateTracker(dim)
        self.reflection_module = ReflectionModule(dim)
        
    def agent_loop(
        self,
        goal_embedded,
        tools,
        episodic_memory=None,
        working_memory=None,
        max_iterations=None,
        enable_reflection=True,
        verbose=True
    ):
        """
        Main agentic loop.
        
        Args:
            goal_embedded: Goal tokens embedded [batch, seq, dim]
            tools: Dictionary of available tools
            episodic_memory: Optional episodic memory module
            working_memory: Optional working memory module
            max_iterations: Max action steps
            enable_reflection: Whether to reflect and self-correct
            verbose: Print progress
        
        Returns:
            dict with success, execution_trace, final_state, reflection
        """
        
        max_iterations = max_iterations or self.max_iterations
        
        # 1. Understand Goal
        goal_structure = self.goal_understanding(goal_embedded)
        goal_encoding = goal_structure['pooled']
        
        if verbose:
            print(f"🎯 Goal complexity: {goal_structure['complexity'].item():.2f}")
        
        # 2. Retrieve Similar Past Episodes
        context = None
        if episodic_memory is not None:
            similar = episodic_memory.retrieve_similar_episodes(goal_encoding, top_k=3)
            if similar is not None:
                context = similar['plans']
                if verbose:
                    print(f"💭 Retrieved {len(similar['plans'])} similar episodes")
        
        # 3. Generate Initial Plan
        initial_plan = self.planning_module.generate_plan(goal_structure, context)
        
        if verbose:
            print(f"\n📋 Initial Plan ({len(initial_plan)} steps)")
        
        # 4. Verify and Optimize Plan
        verification = self.planning_module.verify_plan(initial_plan, goal_structure)
        
        if verbose:
            print(f"✓ Feasibility: {verification['feasibility'].item():.2f}")
        
        if verification['feasibility'].item() < 0.7:
            plan = self.planning_module.optimize_plan(initial_plan)
            if verbose:
                print("🔧 Plan optimized")
        else:
            plan = initial_plan
        
        # 5. Execute Plan with Adaptive Replanning
        current_state = goal_encoding
        execution_trace = []
        working_memory_state = None
        
        iteration = 0
        plan_index = 0
        
        while plan_index < len(plan) and iteration < max_iterations:
            current_step = plan[plan_index]
            
            if verbose:
                print(f"\n🔄 Iteration {iteration + 1}: {current_step['action'][0]}")
            
            # Execute Action
            execution_result = self.action_executor.execute_action(
                current_step,
                tools,
                max_retries=3
            )
            
            success = execution_result['success']
            observation = execution_result['observation']
            
            if verbose:
                status = '✅' if success else '❌'
                print(f"   {status} {'Success' if success else 'Failed'}")
            
            # Update State
            state_update = self.state_tracker.update_state(
                prev_state=current_state,
                observation=observation,
                action=current_step,
                goal=goal_encoding
            )
            
            current_state = state_update['state']
            surprise = state_update['surprise'].item()
            progress = state_update['progress'].item()
            
            if verbose:
                print(f"   Progress: {progress:.1%}")
            
            # Store in Working Memory
            if working_memory is not None and observation.get('embedding') is not None:
                obs_emb = observation['embedding']
                if obs_emb.dim() == 2:
                    obs_emb = obs_emb.unsqueeze(1)
                _, working_memory_state = working_memory(obs_emb, working_memory_state)
            
            # Record Trace
            execution_trace.append({
                'iteration': iteration,
                'action': current_step['action'][0],
                'observation': observation,
                'state': current_state,
                'surprise': surprise,
                'progress': progress,
                'success': success
            })
            
            # Check Goal Achievement
            if progress > 0.95:
                if verbose:
                    print("\n🎉 Goal achieved!")
                break
            
            # Handle Unexpected Outcomes
            if surprise > 0.7 or not success:
                if verbose:
                    print(f"⚠️  Unexpected outcome (surprise={surprise:.2f})")
                
                if enable_reflection and plan_index > 0:
                    # Reflect and Replan
                    reflection = self.reflection_module.reflect(
                        plan=plan[:plan_index+1],
                        execution_trace=execution_trace,
                        goal=goal_encoding,
                        final_state=current_state
                    )
                    
                    if reflection['should_revise']:
                        if verbose:
                            print("🔄 Replanning...")
                        
                        # Generate new plan
                        revised_goal = {
                            'encoded': current_state.unsqueeze(1),
                            'pooled': current_state,
                            'goal_type': goal_structure['goal_type'],
                            'complexity': goal_structure['complexity'],
                            'subgoals': None
                        }
                        new_plan = self.planning_module.generate_plan(revised_goal, None)
                        plan = plan[:plan_index+1] + new_plan
                        
                        if verbose:
                            print(f"   New plan: {len(new_plan)} steps")
                else:
                    if not success:
                        continue  # Retry
            
            plan_index += 1
            iteration += 1
        
        # Final Reflection
        final_reflection = None
        if enable_reflection and len(execution_trace) > 0:
            final_reflection = self.reflection_module.reflect(
                plan=plan,
                execution_trace=execution_trace,
                goal=goal_encoding,
                final_state=current_state
            )
            
            outcome_type = final_reflection['outcome'].argmax(dim=-1).item()
            outcome_names = ['exceeded', 'met', 'failed']
            
            if verbose:
                print(f"\n📊 Final Outcome: {outcome_names[outcome_type]}")
            
            # Store in Episodic Memory if successful
            if episodic_memory is not None and outcome_type <= 1:
                reward = 1.0 if outcome_type == 0 else 0.7
                episodic_memory.store_episode(
                    goal=goal_encoding,
                    plan=plan,
                    outcome=current_state,
                    reward=reward
                )
                if verbose:
                    print(f"💾 Stored in episodic memory (reward={reward})")
        
        # Clear Working Memory
        if working_memory is not None:
            working_memory.clear()
        
        return {
            'success': progress > 0.8,
            'execution_trace': execution_trace,
            'final_state': current_state,
            'iterations': iteration,
            'reflection': final_reflection,
            'progress': progress
        }