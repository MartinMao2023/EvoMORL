from flax.struct import dataclass

from functools import partial
from typing import Any, Tuple, Callable

import flax.linen as nn
import jax
import optax
from jax import numpy as jnp


from buffer import PPOTransition
from networks import QModuleTC
from custom_types import Params, RNGKey, Env, EnvState
from flax.struct import PyTreeNode
# from losses import bce_loss



class PPOTrainingState(PyTreeNode):
    """Contains training state for the learner."""

    critic_params: Params
    policy_params: Params

    critic_opt_state: optax.OptState
    policy_opt_state: optax.OptState



class MO_PPO:

    def __init__(
        self,
        env: Env,
        policy_network: nn.Module,
        critic_network: nn.Module,
        policy_learnng_rate: float = 5e-4,
        critic_learning_rate: float = 5e-4,
        clip_ratio: float = 0.5,
        entropy_gain: float = 0.0,
        dim: int=4,
        fixed_std: bool = True,
        include_last_action_in_obs: bool = False,
        ):

        self._env = env
        self._include_last_action_in_obs = include_last_action_in_obs
        self._entropy_gain = entropy_gain
        self._policy_network = policy_network
        self._critic_network = critic_network
        self._dim = dim
        if clip_ratio > 0:
            self._clip_log_ratio = jnp.log(1 + clip_ratio)
        else:
            raise(ValueError("invalid clip ratio"))

        self._policy_optimizer = optax.adam(
            learning_rate=policy_learnng_rate,
        )
        self._critic_optimizer = optax.adam(
            learning_rate=critic_learning_rate,
        )

        def critic_loss_fn(
            critic_params: Params,
            transitions: PPOTransition,
        ) -> float:

            estimated_v = critic_network.apply(critic_params, transitions.obs, transitions.preferences)
            weights = 1 / (100 - 99*transitions.weights)

            return jnp.mean(jnp.square(estimated_v - transitions.td_lambda_returns) * weights)
        
        self._critic_loss_fn = critic_loss_fn

        if fixed_std:
            def policy_loss_fn(
                policy_params: Params,
                transitions: PPOTransition,
            ) -> float:
                
                gaes = transitions.gaes # batch x 1

                gae_std = jnp.std(gaes) 
                rescaled_gaes = gaes / (1e-6 + gae_std)
                action_mean, _ = policy_network.apply(policy_params, transitions.obs, transitions.preferences)
                old_action_variance = jnp.exp(2*transitions.action_log_std)

                old_distance = jax.lax.stop_gradient(
                    jnp.sum(jnp.square(transitions.action_noises) / old_action_variance, axis=-1, keepdims=True)
                    )
                new_distance = jax.lax.stop_gradient(
                    jnp.sum(jnp.square(action_mean - transitions.actions) / old_action_variance, axis=-1, keepdims=True)
                )
                distance = jnp.where(rescaled_gaes > 0, old_distance - new_distance, new_distance - old_distance)
                scale = jnp.exp(2*jnp.mean(transitions.action_log_std, axis=-1, keepdims=True))

                loss = jnp.where(
                    distance < 2 * self._clip_log_ratio,
                    -rescaled_gaes * scale * jnp.exp(
                        - 0.5*jnp.sum(jnp.square(action_mean - transitions.actions)/old_action_variance, axis=-1, keepdims=True)
                        + 0.5*jnp.sum(jnp.square(transitions.action_noises)/old_action_variance, axis=-1, keepdims=True)
                        ) ,
                    0.0
                )

                return jnp.mean(loss * transitions.weights)

        else:
            def policy_loss_fn(
                policy_params: Params,
                transitions: PPOTransition,
            ) -> float:
                
                gaes = transitions.gaes # batch x 1

                gae_std = jnp.std(gaes) 
                rescaled_gaes = gaes / (1e-6 + gae_std)
                action_mean, action_std = policy_network.apply(policy_params, transitions.obs, transitions.preferences)

                old_action_variance = jnp.exp(2*transitions.action_log_std) + 1e-6
                new_action_variance = action_std + 1e-6

                old_distance = jax.lax.stop_gradient(
                    jnp.sum(jnp.square(transitions.action_noises) / old_action_variance, axis=-1, keepdims=True)
                    )
                new_distance = jax.lax.stop_gradient(
                    jnp.sum(jnp.square(action_mean - transitions.actions) / new_action_variance, axis=-1, keepdims=True)
                )
                distance = jnp.where(rescaled_gaes > 0, old_distance - new_distance, new_distance - old_distance)
                scale = jnp.exp(2*jnp.mean(transitions.action_log_std, axis=-1, keepdims=True))

                loss = jnp.where(
                    distance < 2 * self._clip_log_ratio,
                    -rescaled_gaes * scale * jnp.exp(
                        - 0.5*jnp.sum(jnp.square(action_mean - transitions.actions)/new_action_variance, axis=-1, keepdims=True)
                        + 0.5*jnp.sum(jnp.square(transitions.action_noises)/old_action_variance, axis=-1, keepdims=True)
                        ) - self._entropy_gain * jnp.mean(jnp.log(action_std + 1e-6), axis=-1, keepdims=True),
                    0.0
                )

                return jnp.mean(loss * transitions.weights)

        self._policy_loss_fn = policy_loss_fn


    def init(
        self, 
        key: RNGKey,
    ) -> PPOTrainingState:
        
        if self._include_last_action_in_obs:
            observation_size = self._env.observation_size + self._env.action_size
        else:
            observation_size = self._env.observation_size

        fake_obs = jnp.zeros(shape=(observation_size,))
        fake_preference = jnp.zeros(shape=(self._dim,))

        key, subkey = jax.random.split(key)
        policy_params = self._policy_network.init(subkey, fake_obs, fake_preference)
        policy_opt_state = self._policy_optimizer.init(policy_params)

        key, subkey = jax.random.split(key)
        critic_params = self._critic_network.init(subkey, fake_obs, fake_preference)
        critic_opt_state = self._critic_optimizer.init(critic_params)
        
        training_state = PPOTrainingState(
            critic_params=critic_params,
            policy_params=policy_params,
            critic_opt_state=critic_opt_state,
            policy_opt_state=policy_opt_state,
            )
        
        return training_state

    
    def train(
        self, 
        training_state: PPOTrainingState, 
        transitions: PPOTransition, 
        critic_epoch: int, 
        policy_epoch: int,
    ) -> PPOTrainingState:
        """
        This function cannot be Jit-complied.
        """

        policy_params = training_state.policy_params
        critic_params = training_state.critic_params
        policy_opt_state = training_state.policy_opt_state
        critic_opt_state = training_state.critic_opt_state

        for i in range(critic_epoch):
            critic_params, critic_opt_state = self.train_critic(
                critic_params,
                critic_opt_state,
                transitions,
            )
        
        for i in range(policy_epoch):
            policy_params, policy_opt_state = self.train_policy(
                policy_params,
                policy_opt_state,
                transitions,
            )

        training_state = training_state.replace(
            policy_params=policy_params,
            critic_params=critic_params,
            policy_opt_state=policy_opt_state,
            critic_opt_state=critic_opt_state,
        )

        return training_state


    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def train_policy(
        self,
        policy_params: Params,
        policy_opt_state: optax.OptState,
        transitions: PPOTransition,
        ):
        """
        train policy network
        """

        def scan_train_policy(carry, transition_data):
            current_policy_params, current_policy_opt_state = carry

            policy_gradient = jax.grad(self._policy_loss_fn)(
                current_policy_params,
                transition_data,
                )

            policy_updates, new_policy_opt_state = self._policy_optimizer.update(
                policy_gradient, current_policy_opt_state)
            new_policy_params = optax.apply_updates(current_policy_params, policy_updates)
            
            return (new_policy_params, new_policy_opt_state), None
        

        (final_policy_params, final_policy_opt_state), _ = jax.lax.scan(
            scan_train_policy,
            (policy_params, policy_opt_state),
            transitions,
        )
        
        return final_policy_params, final_policy_opt_state


    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def train_critic(
        self,
        critic_params: Params,
        critic_opt_state: optax.OptState,
        transitions: PPOTransition,
    ) -> PPOTrainingState:
        """
        train critic network
        """
        
        def scan_train_critic(carry, transition_data):
            current_critic_params, current_critic_opt_state = carry

            critic_gradient = jax.grad(self._critic_loss_fn)(
                current_critic_params,
                transition_data,
                )

            critic_updates, new_critic_opt_state = self._critic_optimizer.update(
                critic_gradient, current_critic_opt_state)
            new_critic_params = optax.apply_updates(current_critic_params, critic_updates)
            
            return (new_critic_params, new_critic_opt_state), None
        

        (final_critic_params, final_critic_opt_state), _ = jax.lax.scan(
            scan_train_critic,
            (critic_params, critic_opt_state),
            transitions,
        )
        
        return final_critic_params, final_critic_opt_state
    


    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def calculate_v(
        self,
        critic_params: Params, 
        obs: jnp.ndarray,
        preferences: jnp.ndarray,
    ) -> jnp.ndarray:
        
        def scan_calculate_v(
            carry: jnp.ndarray,
            data: jnp.ndarray,
        ) -> Tuple[None, jnp.ndarray]:
            
            preference = carry
            ob = data
            v_value = self._critic_network.apply(critic_params, ob, preference)

            return preference, v_value

        _, v_values = jax.lax.scan(
            jax.vmap(scan_calculate_v),
            preferences,
            obs,
            )
        
        return v_values
    


def staged_ppo_training(
    policy_params: Params,
    residual_params: Params,
    critic_params: Params,
    initial_state: EnvState,
    vec_env: int = 64,
    rollout_length: int = 128,
    stage_iter: int = 16,
) -> tuple[Params, Params, PPOTransition, jnp.ndarray]:
    
    """
    1) rollout from the starting state

    2) calculate GAE

    3) update policy

    4) update residual netweork

    5) reset to starting state and loop m times

    
    for final iteration:
    
    1) rollout from the starting state
    
    2) calculate optimisitic GAE

    3) update policy (optional)

    4) update residual netweork

    5) calculate the average GAEs over the critic network (without residue)


    returns:
        policy, residual_params, transitions, PA

    """



    def scan_training():
        pass



    

    return




