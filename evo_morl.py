from flax.struct import dataclass

from functools import partial
from typing import Any, Tuple, Callable

import flax.linen as nn
import jax
import optax
from jax import numpy as jnp


from buffer import PPOTransition as MO_PPO_Transition
from networks import QModuleTC
from custom_types import Params, RNGKey, Env, EnvState, EmitterState, ExtraScores
from flax.struct import PyTreeNode
from ga_repertoire import GARepertoire
# from buffer import PPOTransition
from mutation_operators import isoline_variation
import functools




@dataclass
class EvoMORLConfig:
    """Configuration for EvoMORL emitter"""
    # NeuroEvolution configs
    ga_popsize: int = 256
    ai_popsize: int = 256
    iso_sigma: float = 0.005
    line_sigma: float = 0.05

    # General RL configs
    gamma_discount: float = 0.99
    lambda_discount: float = 0.95


    # PPO configs
    num_envs: int = 64 # num of parallel agents for each policy
    rollout_length: int = 64 # horizon for each PPO rollout
    ppo_batchsize: int = 256 # mini-batch size

    policy_epochs: int = 8
    critic_epochs: int = 8
    critic_learning_rate: float = 5e-4
    policy_learning_rate: float = 5e-4

    # Other configs
    num_eval: int = 16 # parallel agents during evaluation rollout 

    # estimator_learning_rate: float = 3e-4
    # policy_learning_rate: float = 3e-4








class EvoMORL:

    def __init__(self, config: EvoMORLConfig):

        self._config = config

        self._variation_fn = functools.partial(
            isoline_variation, 
            iso_sigma=config.iso_sigma, 
            line_sigma=config.line_sigma,
        )

        pass



    def init(
        self, 
        key: RNGKey,
    ):
        pass

    
    def train(
        self, 
        emitter_state: EmitterState, 
        archive: GARepertoire, 
        key: RNGKey,
    ) -> Tuple[Tuple[EmitterState, GARepertoire, RNGKey], ExtraScores]:
        


        key, subkey = jax.random.split(key)

        new_emitter_state = emitter_state

        genotypes = None
        fitnesses = None
        new_archive = archive.add(genotypes, fitnesses)

        """
        ExtraScores: 
            average policy advantages, change of PA after fine tuning, contributions, HyperVolumn
        
        """


        return (new_emitter_state, new_archive, key), None



    def evaluate(
        self,
        policy_params: Params,
        starting_states: EnvState,
        keys: RNGKey,
        rollout_length: int,
    ) -> Tuple[Tuple[EnvState, RNGKey], MO_PPO_Transition]:
        """
        This function is not Jit-compatible by default.
        To make it Jit-able, wrap it with partial(evaluate, rollout_length=n).
        """
        pass


    def emit_ai(
        self, 
        emitter_state: EmitterState, 
        preferences: jnp.ndarray
    ) -> Params:
        
        pass


    def emit_ga(
        self, 
        archive: GARepertoire,
        key: RNGKey,
    ) -> Params:
        
        ga_popsize = self._config.ga_popsize
        
        x1, key = archive.sample(key, ga_popsize)
        x2, key = archive.sample(key, ga_popsize)
        ga_offsprings, key = self._variation_fn(x1, x2, key)
        
        return ga_offsprings, key



    def relocate(self):
        pass


    def fine_tune(self):
        pass
    

    def distillate(self):
        pass






