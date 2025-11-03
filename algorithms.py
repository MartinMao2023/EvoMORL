from flax.struct import dataclass

from functools import partial
from typing import Any, Tuple, Callable

import flax.linen as nn
import jax
import optax
from jax import numpy as jnp


from buffer import PPOTransition
# from networks import QModuleTC
from custom_types import Params, RNGKey, Env, EnvState
from flax.struct import PyTreeNode
from utils import shuffle_transitions


@dataclass
class PPOConfigs:
    policy_learnng_rate: float = 5e-4
    critic_learning_rate: float = 5e-4
    clip_ratio: float = 0.5
    entropy_gain: float = 0.0
    gamma: float = 0.99
    td_lambda: float = 0.95
    rollout_length: int = 128
    mini_batch_size: int = 256
    critic_epoch: int = 4
    policy_epoch: int = 4
    learnable_std: bool = False

    # not implemented yet
    initial_std: jnp.ndarray = 0.2
    std_decay_rate: float = 5e-5
    mini_std: jnp.ndarray = 0.05



class PPOTrainingState(PyTreeNode):
    """Contains training state for the learner."""

    critic_params: Params
    policy_params: Params

    critic_opt_state: optax.OptState
    policy_opt_state: optax.OptState




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
        self.configs = ppo_configs
        self._include_last_action_in_obs = include_last_action_in_obs

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


        #######################################################
        #                                                     #
        #            build rollout functions                  #
        #                                                     #
        #######################################################


        def critic_loss_fn(
            critic_params: Params,
            transitions: PPOTransition,
        ) -> float:

            estimated_v = critic_network.apply(critic_params, transitions.obs)
            weights = 1 / (100 - 99*transitions.weights)

            return jnp.mean(jnp.square(estimated_v - transitions.td_lambda_returns) * weights)
        
        self._critic_loss_fn = critic_loss_fn

        if ppo_configs.learnable_std:
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
                        ) - self.configs.entropy_gain * jnp.mean(jnp.log(action_std + 1e-6), axis=-1, keepdims=True),
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

                action_mean, _ = policy_network.apply(policy_params, transitions.obs)
                action_variance = jnp.exp(2*transitions.action_log_std)

                old_distance = jax.lax.stop_gradient(
                    jnp.sum(jnp.square(transitions.action_noises) / action_variance, axis=-1, keepdims=True)
                    )
                new_distance = jax.lax.stop_gradient(
                    jnp.sum(jnp.square(action_mean - transitions.actions) / action_variance, axis=-1, keepdims=True)
                )
                distance = jnp.where(rescaled_gaes > 0, old_distance - new_distance, new_distance - old_distance)
                scale = jnp.exp(2*jnp.mean(transitions.action_log_std, axis=-1, keepdims=True))

                loss = jnp.where(
                    distance < 2 * self._clip_log_ratio,
                    -rescaled_gaes * scale * jnp.exp(
                        - 0.5*jnp.sum(jnp.square(action_mean - transitions.actions)/action_variance, axis=-1, keepdims=True)
                        + 0.5*jnp.sum(jnp.square(transitions.action_noises)/action_variance, axis=-1, keepdims=True)
                        ),
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


    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def state_update(
        self, 
        training_state: PPOTrainingState, 
        transitions: PPOTransition, 
    ) -> PPOTrainingState:
        """
        This function can now be Jit-complied.
        """

        (critic_params, critic_opt_state), _ = jax.lax.scan(
            lambda x, _: partial(self.train_critic, transitions=transitions)(x),
            (training_state.critic_params, training_state.critic_opt_state),
            length=self.configs.critic_epoch,
        )

        (policy_params, policy_opt_state), _ = jax.lax.scan(
            lambda x, _: partial(self.train_policy, transitions=transitions)(x),
            (training_state.policy_params, training_state.policy_opt_state),
            length=self.configs.policy_epoch,
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
        carry: Tuple[Params, optax.OptState],
        transitions: PPOTransition,
        ):
        """
        train policy network
        """

        policy_params, policy_opt_state = carry

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
        
        return (final_policy_params, final_policy_opt_state), None



    # Ultilities
    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def train_critic(
        self,
        carry: Tuple[Params, optax.OptState],
        transitions: PPOTransition,
    ) -> PPOTrainingState:
        """
        train critic network
        """
        critic_params, critic_opt_state = carry
        
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
        
        return (final_critic_params, final_critic_opt_state), None
    


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
    

    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def process_gaes(
        self,
        gaes: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Flter the extreme values of GAEs and subtract the mean if needed
        """
        gae_mean = jnp.mean(gaes)
        gae_std = jnp.std(gaes)
        mask = jnp.abs(gaes - gae_mean) < 3*gae_std
        filtered_mean = jnp.mean(gaes, where=mask)
        filted_std = jnp.std(gaes, where=mask)
        gaes = jnp.clip(gaes, filtered_mean - 8*filted_std, filtered_mean + 8*filted_std)
        offset = jnp.where(filtered_mean < 0, -filtered_mean, 0.0)
        
        return gaes + offset
    

    def train(
        self,
        starting_states: EnvState,
        training_state: PPOTrainingState,
        iterations: int,
        key: RNGKey,
    ) -> Tuple[PPOTrainingState, EnvState]:
        
        # use training state to conduct rollout
        states = starting_states
        # calucalte gaes

        # process gaes

        # collect data into transitions

        # shuffle and reshape transitions
        transitions = shuffle_transitions(subkey, transitions)
        transitions = jax.tree.map(
            lambda x: jnp.reshape(
                x,
                (
                    -1,
                    self.configs.mini_batch_size,
                    *x.shape[1:],
                ),
            ),
            transitions)
        
        new_training_state = self.state_update(training_state, transitions)

        return new_training_state, states

