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


@dataclass
class PPOConfigs:
    policy_learnng_rate: float = 5e-4
    critic_learning_rate: float = 5e-4
    clip_ratio: float = 0.5
    entropy_gain: float = 0.0
    fixed_std: bool = True






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
        entropy_gain: float = 0.0,
        fixed_std: bool = True,
        include_last_action_in_obs: bool = False,
        ):

        self._env = env
        self._include_last_action_in_obs = include_last_action_in_obs
        self._entropy_gain = entropy_gain
        self._policy_network = policy_network
        self._critic_network = critic_network

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

            estimated_v = critic_network.apply(critic_params, transitions.obs)
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

                action_gradient = rescaled_gaes * transitions.action_noises
                delta_action_norm = jnp.sqrt(jnp.sum(jnp.square(transitions.action_noises), axis=-1, keepdims=True)) # batch x 1
                action_mean, _ = policy_network.apply(policy_params, transitions.obs)

                mu_d = 2 * mean_kl_threshold * jnp.mean(jnp.exp(transitions.action_log_std), axis=-1, keepdims=True)
                # mu_d = 0.1
                k = jnp.abs(rescaled_gaes) * jnp.clip(
                    delta_action_norm / mu_d, 
                    min=1.0,
                )

                q_values = jnp.sum(action_mean * action_gradient, axis=-1, keepdims=True)
                old_action_mean = transitions.actions - transitions.action_noises

                old_distance = jax.lax.stop_gradient(
                    jnp.sum(jnp.square(old_action_mean - transitions.actions), axis=-1, keepdims=True)
                    )
                new_distance = jax.lax.stop_gradient(
                    jnp.sum(jnp.square(action_mean - transitions.actions), axis=-1, keepdims=True)
                )

                squared_deviation = jnp.sum(jnp.square(action_mean - old_action_mean), axis=-1, keepdims=True) # batch x 1
                positive_loss = k * squared_deviation - 2 * q_values

                old_action_variance = jnp.exp(2*transitions.action_log_std)

                scale = jax.lax.stop_gradient(
                    jnp.exp(
                    - 0.5*jnp.sum(jnp.square(action_mean - transitions.actions)/old_action_variance, axis=-1, keepdims=True)
                    + 0.5*jnp.sum(jnp.square(transitions.action_noises)/old_action_variance, axis=-1, keepdims=True)
                    )
                )
                scale = jnp.clip(scale, 0.5, 2)

                negative_loss = jnp.where(
                    new_distance - old_distance > mu_d**2,
                    0.0,
                    rescaled_gaes * jnp.sum(jnp.square(action_mean - transitions.actions), axis=-1, keepdims=True),
                ) 


                policy_losses = scale * jnp.where(
                    gaes > 0,
                    positive_loss,
                    negative_loss,
                    # negative_reg,
                )

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
        
        if self._include_last_action_in_obs:
            observation_size = self._env.observation_size + self._env.action_size
        else:
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





class PPO:

    def __init__(
        self,
        env: Env,
        policy_network: nn.Module,
        critic_network: nn.Module,
        ppo_configs: PPOConfigs,
        include_last_action_in_obs: bool = False,
        ):

        self._env = env
        self._include_last_action_in_obs = include_last_action_in_obs
        self._entropy_gain = ppo_configs.entropy_gain
        self._policy_network = policy_network
        self._critic_network = critic_network
        if ppo_configs.clip_ratio > 0:
            self._clip_log_ratio = jnp.log(1 + ppo_configs.clip_ratio)
        else:
            raise(ValueError("invalid clip ratio"))

        self._policy_optimizer = optax.adam(
            learning_rate=ppo_configs.policy_learnng_rate,
        )
        self._critic_optimizer = optax.adam(
            learning_rate=ppo_configs.critic_learning_rate,
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
                
                gaes = transitions.gaes # batch x 1

                gae_std = jnp.std(gaes) 
                rescaled_gaes = gaes / (1e-6 + gae_std)
                action_mean, _ = policy_network.apply(policy_params, transitions.obs)

                old_action_variance = jnp.exp(2*transitions.action_log_std)
                # new_action_variance = jnp.exp(2*action_log_std)

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
                action_mean, action_std = policy_network.apply(policy_params, transitions.obs)

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
    


