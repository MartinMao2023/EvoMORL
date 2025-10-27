from flax.struct import dataclass

from functools import partial
from typing import Any, Tuple, Callable

import flax.linen as nn
import jax
import optax
from jax import numpy as jnp


from buffer import PPOTransition
from networks import QModuleTC
from custom_types import Params, RNGKey, Env
from flax.struct import PyTreeNode
# from losses import bce_loss



class PPOTrainingState(PyTreeNode):
    """Contains training state for the learner."""

    critic_params: Params
    policy_params: Params

    critic_opt_state: optax.OptState
    policy_opt_state: optax.OptState



class QuaraticPPO:

    def __init__(
        self,
        env: Env,
        policy_network: nn.Module,
        critic_network: nn.Module,
        policy_learnng_rate: float = 5e-4,
        critic_learning_rate: float = 5e-4,
        mean_kl_threshold: float = 0.5,
        log_std_kl_threshold: float = 0.5,
        fixed_std: bool = True
        ):

        self._env = env

        self._policy_network = policy_network
        self._critic_network = critic_network

        self._policy_optimizer = optax.adam(
            learning_rate=policy_learnng_rate,
            # weight_decay=0.001,
        )
        self._critic_optimizer = optax.adam(
            learning_rate=critic_learning_rate,
            # weight_decay=0.001,
        )

        def critic_loss_fn(
            critic_params: Params,
            transitions: PPOTransition,
        ) -> float:

            estimated_v = critic_network.apply(critic_params, transitions.obs)
            weights = 1 / (100 - 99*transitions.weights)

            return jnp.mean(jnp.square(estimated_v - transitions.td_lambda_returns) * weights)
        
        self._critic_loss_fn = critic_loss_fn

        if fixed_std:
            def policy_loss_fn(
                policy_params: Params,
                transitions: PPOTransition,
            ) -> float:
                
                advantages = transitions.gaes # batch x 1

                gae_std = jnp.std(advantages) 
                gae_mean = jnp.mean(advantages)
                gaes = jnp.clip(advantages, gae_mean - 4 * gae_std, gae_mean + 4 * gae_std)

                # normalized_advantages = (gaes - gae_mean) / (1e-6 + gae_std)
                rescaled_advantages = gaes / (1e-6 + gae_std)


                action_gradient = rescaled_advantages * transitions.action_noises
                # normalized_action_gradient = normalized_advantages * transitions.action_noises

                gradient_norm = jnp.sqrt(jnp.sum(jnp.square(action_gradient), axis=-1, keepdims=True)) # batch x 1

                action_mean, action_log_std = policy_network.apply(policy_params, transitions.obs)

                k = jax.lax.stop_gradient(gradient_norm / (mean_kl_threshold * 2 * jnp.exp(action_log_std)))

                q_values = jnp.sum(action_mean * action_gradient, axis=-1, keepdims=True)

                old_action_mean = transitions.actions - transitions.action_noises
                policy_losses = k * jnp.sum(jnp.square(action_mean - old_action_mean), axis=-1, keepdims=True) - 2 * q_values

                return jnp.mean(policy_losses * transitions.weights)

        else:
            def policy_loss_fn(
                policy_params: Params,
                transitions: PPOTransition,
            ) -> float:
                
                gaes = transitions.gaes # batch x 1
                gae_std = jnp.std(gaes) 
                # gae_mean = jnp.mean(gaes)
                rescaled_gaes = gaes / (1e-6 + gae_std)

                old_log_std = transitions.action_log_std # batch x action_dim


                v = transitions.action_noises * jnp.exp(-2*old_log_std)
                action_mean, action_log_std = policy_network.apply(policy_params, transitions.obs)
                action_variance_sg = jax.lax.stop_gradient(jnp.exp(2*action_log_std))
                old_action_mean = transitions.actions - transitions.action_noises

                mean_grad = rescaled_gaes * v
                mean_k = jnp.abs(rescaled_gaes) * jnp.sqrt(jnp.sum(
                    v**2 * action_variance_sg, 
                    axis=-1, keepdims=True) / (mean_kl_threshold * 2))

                log_std_grad = rescaled_gaes * (v * transitions.action_noises  - 1)
                log_std_k  = jnp.abs(rescaled_gaes) * jnp.sqrt(jnp.sum((v * transitions.action_noises  - 1)**2, axis=-1, keepdims=True) / log_std_kl_threshold)

                q_diff = jnp.sum(action_mean * mean_grad, axis=-1, keepdims=True) + jnp.sum(action_log_std * log_std_grad, axis=-1, keepdims=True)

                mean_penalty = 0.5 * mean_k * jnp.sum(jnp.square(action_mean - old_action_mean) / action_variance_sg, axis=-1, keepdims=True)
                
                log_std_penalty = 0.5 * log_std_k * jnp.sum(jnp.square(action_log_std - old_log_std), axis=-1, keepdims=True)

                
                policy_losses = mean_penalty + log_std_penalty - q_diff - action_log_std * 0.01

                return jnp.mean(policy_losses)

        self._policy_loss_fn = policy_loss_fn


    def init(
        self, 
        key: RNGKey,
    ) -> PPOTrainingState:
        
        observation_size = self._env.observation_size

        fake_obs = jnp.zeros(shape=(observation_size,))

        key, subkey = jax.random.split(key)
        policy_params = self._policy_network.init(subkey, obs=fake_obs)
        policy_opt_state = self._policy_optimizer.init(policy_params)

        key, subkey = jax.random.split(key)
        critic_params = self._critic_network.init(subkey, obs=fake_obs)
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
    ) -> jnp.ndarray:
        
        def scan_calculate_v(
            ob: jnp.ndarray
        ) -> Tuple[None, jnp.ndarray]:
            
            v_value = self._critic_network.apply(critic_params, ob)

            return None, v_value

        _, v_values = jax.lax.scan(
            lambda _, x: jax.vmap(scan_calculate_v)(x),
            None,
            obs,   
            )
        
        return v_values





