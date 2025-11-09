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
    clip_ratio: float = 0.2
    entropy_gain: float = 0.0
    discount: float = 0.99
    td_lambda_discount: float = 0.95
    rollout_length: int = 128
    vec_env: int = 64
    mini_batch_size: int = 256
    critic_epochs: int = 4
    policy_epochs: int = 4
    learnable_std: bool = False

    initial_std: jnp.ndarray = 0.2
    std_decay_rate: float = 5e-5
    min_std: jnp.ndarray = 0.05



class PPOTrainingState(PyTreeNode):
    """Contains training state for the learner."""

    critic_params: Params
    policy_params: Params

    critic_opt_state: optax.OptState
    policy_opt_state: optax.OptState

    current_std: jnp.ndarray



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
        
        if self._include_last_action_in_obs:
            self._observation_size = env.observation_size + env.action_size
        else:
            self._observation_size = env.observation_size
        

        self._policy_optimizer = optax.adam(
            learning_rate=ppo_configs.policy_learnng_rate,
        )
        self._critic_optimizer = optax.adam(
            learning_rate=ppo_configs.critic_learning_rate,
        )

        if include_last_action_in_obs:
            self._build_obs = lambda x, y: jnp.concatenate([x, y], axis=-1)
        else:
            self._build_obs = lambda x, y: x


        @jax.jit
        def rollout_fn(
            policy_params: Params,
            starting_states: EnvState,
            last_action_means: jnp.ndarray,
            keys: RNGKey,
            std: jnp.ndarray,
            ) -> PPOTransition:

            def play_step_fn(
                carry: Tuple[EnvState, jnp.ndarray, int, RNGKey],
                ) -> Tuple[Tuple, PPOTransition]:
                
                state, last_action_mean, key = carry
                key, subkey = jax.random.split(key)
                obs = self._build_obs(state.obs, last_action_mean)

                action_mean, action_std = policy_network.apply(policy_params, obs)
                action_std = jax.lax.select(
                    self.configs.learnable_std, 
                    action_std, 
                    std*jnp.ones_like(action_std)
                    )

                candidate_action_noise = action_std * jax.random.normal(subkey, action_mean.shape)
                action = jnp.clip(action_mean + candidate_action_noise, -1.0, 1.0)
                action_noise = action - action_mean
                action_log_std = jnp.log(action_std + 1e-6)

                next_state = env.step(state, action)
                # rewards = jnp.array([
                #     state.metrics["reward_forward"] + 1, 
                #     state.metrics["reward_ctrl"] + 0.25, 
                #     state.pipeline_state.x.pos[0, 2],
                #     1 - 2.5*jnp.mean(jnp.square(action_mean - last_action_mean)) # zero'th order smoothness
                #     ])
                rewards = jnp.ones((1,))*next_state.reward

                transition = PPOTransition(
                    obs=obs,
                    actions=action,
                    action_noises=action_noise,
                    action_log_std=action_log_std,
                    rewards=rewards,
                    preferences=jnp.zeros_like(rewards),
                    td_lambda_returns=jnp.zeros((1,)),
                    baselines=jnp.zeros((1,)),
                    gaes=jnp.zeros((1,)),
                    dones=next_state.done,
                    truncations=0.0,
                    weights=jnp.zeros((1,)),
                    )

                return (next_state, action_mean, key), transition

            final_carry, transitions = jax.lax.scan(
                lambda x, _: jax.vmap(play_step_fn)(x),
                (starting_states, last_action_means, keys),
                length=ppo_configs.rollout_length,
            )
            
            final_states, final_action, _= final_carry

            return final_states, final_action, transitions
        

        self._rollout_fn = rollout_fn


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
        
        fake_obs = jnp.zeros(shape=(self._observation_size,))

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
            current_std=self.configs.initial_std,
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
            length=self.configs.critic_epochs,
        )

        (policy_params, policy_opt_state), _ = jax.lax.scan(
            lambda x, _: partial(self.train_policy, transitions=transitions)(x),
            (training_state.policy_params, training_state.policy_opt_state),
            length=self.configs.policy_epochs,
        )

        current_std = training_state.current_std - self.configs.std_decay_rate
        
        training_state = training_state.replace(
            policy_params=policy_params,
            critic_params=critic_params,
            policy_opt_state=policy_opt_state,
            critic_opt_state=critic_opt_state,
            current_std=jnp.clip(current_std, min=self.configs.min_std),
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
    

    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def train(
        self,
        starting_states: EnvState,
        last_action_means: jnp.ndarray,
        training_state: PPOTrainingState,
        key: RNGKey,
    ) -> Tuple[PPOTrainingState, EnvState]:
        """
        Perform one iteration of PPO update
        
        """
        
        # use training state to conduct rollout

        policy_params = training_state.policy_params
        critic_params = training_state.critic_params
        current_std = training_state.current_std

        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, num=self.configs.vec_env)
        (
            states, last_actions, transitions
            ) = self._rollout_fn(policy_params, starting_states, last_action_means, subkeys, current_std)

        last_obs = self._build_obs(states.obs, last_actions)
        final_v_value = self._critic_network.apply(critic_params, last_obs)
        v_values = self.calculate_v(critic_params, transitions.obs)

        td_lambda_returns, weights = self.calculate_td_lambda_returns(
            final_v_value,
            v_values, 
            transitions.rewards,
            jnp.clip(1 - transitions.truncations - transitions.dones, min=0.0),
            ) # rollout x parallelize

        # mean_change = jnp.average(td_lambda_returns - v_values, axis=0, weights=weights, keepdims=True) # 1 x parallelize
        # td_lambda_returns = td_lambda_returns + (1 - weights) * mean_change # corrected td_lambda_returns

        gaes = td_lambda_returns - v_values
        gaes = self.process_gaes(gaes)

        transitions = transitions.replace(
            td_lambda_returns=td_lambda_returns,
            gaes=gaes,
            weights=weights,
            )
        
        key, subkey = jax.random.split(key)

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

        return (states, last_actions, new_training_state, key), transitions



    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def calculate_td_lambda_returns(
        self,
        final_v_value: jnp.ndarray,
        v_values: jnp.ndarray, 
        rewards: jnp.ndarray,
        masks: jnp.ndarray,
    ) -> jnp.ndarray:
        
        discount = self.configs.discount
        td_lambda_discount = self.configs.td_lambda_discount


        def scan_calculate_td_lambda(
            carry: jnp.ndarray, 
            data: Tuple[jnp.ndarray, jnp.ndarray],
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            
            (last_td_lambda_value, last_value, last_weight) = carry
            reward, v_value, mask = data
            current_td_lambda_value = reward + mask * discount * (
                    (1 - td_lambda_discount) * last_value + td_lambda_discount * last_td_lambda_value
                )
            weight = discount * td_lambda_discount * (last_weight - 1) * mask + 1

            return (current_td_lambda_value, v_value, weight), (current_td_lambda_value, weight)
            
        _, (td_lambda_values, weights) = jax.lax.scan(
            jax.vmap(scan_calculate_td_lambda),
            (final_v_value, final_v_value, jnp.zeros_like(final_v_value)),
            (rewards, v_values, masks),
            reverse=True,
        ) # length x batch x d

        return td_lambda_values, weights




class MOPPO:
    def __init__(
        self,
        env: Env,
        policy_network: nn.Module,
        critic_network: nn.Module,
        ppo_configs: PPOConfigs,
        sample_fn: Callable,
        include_last_action_in_obs: bool = True,
        ):

        self._env = env
        self._sample_fn = sample_fn
        self.configs = ppo_configs
        self._include_last_action_in_obs = include_last_action_in_obs

        self._policy_network = policy_network
        self._critic_network = critic_network

        if ppo_configs.clip_ratio > 0:
            self._clip_log_ratio = jnp.log(1 + ppo_configs.clip_ratio)
        else:
            raise(ValueError("invalid clip ratio"))
        
        if self._include_last_action_in_obs:
            self._observation_size = env.observation_size + env.action_size
        else:
            self._observation_size = env.observation_size
        

        self._policy_optimizer = optax.adam(
            learning_rate=ppo_configs.policy_learnng_rate,
        )
        self._critic_optimizer = optax.adam(
            learning_rate=ppo_configs.critic_learning_rate,
        )

        if include_last_action_in_obs:
            self._build_obs = lambda x, y: jnp.concatenate([x, y], axis=-1)
        else:
            self._build_obs = lambda x, y: x


        @jax.jit
        def rollout_fn(
            policy_params: Params,
            starting_states: EnvState,
            last_action_means: jnp.ndarray,
            preferences: jnp.ndarray,
            keys: RNGKey,
            std: jnp.ndarray,
            ) -> PPOTransition:

            def play_step_fn(
                carry: Tuple[EnvState, jnp.ndarray, int, RNGKey],
                ) -> Tuple[Tuple, PPOTransition]:
                
                state, last_action_mean, preference, key = carry
                key, subkey = jax.random.split(key)
                obs = self._build_obs(state.obs, last_action_mean)

                action_mean, action_std = policy_network.apply(policy_params, obs, preference)
                action_std = jax.lax.select(
                    self.configs.learnable_std, 
                    action_std, 
                    std*jnp.ones_like(action_std)
                    )

                candidate_action_noise = action_std * jax.random.normal(subkey, action_mean.shape)
                action = jnp.clip(action_mean + candidate_action_noise, -1.0, 1.0)
                action_noise = action - action_mean
                action_log_std = jnp.log(action_std + 1e-6)

                next_state = env.step(state, action)
                
                rewards = jnp.array([
                    state.metrics["reward_forward"] + 1, 
                    state.metrics["reward_ctrl"] + 0.25, 
                    state.pipeline_state.x.pos[0, 2],
                    1 - 2.5*jnp.mean(jnp.square(action_mean - last_action_mean)) # zero'th order smoothness
                    ])
                # rewards = next_state.reward

                transition = PPOTransition(
                    obs=obs,
                    actions=action,
                    action_noises=action_noise,
                    action_log_std=action_log_std,
                    rewards=rewards,
                    preferences=preference,
                    td_lambda_returns=jnp.zeros((1,)),
                    baselines=jnp.zeros((1,)),
                    gaes=jnp.zeros((1,)),
                    dones=next_state.done,
                    truncations=0.0,
                    weights=jnp.zeros((1,)),
                    )

                return (next_state, action_mean, preference, key), transition

            final_carry, transitions = jax.lax.scan(
                lambda x, _: jax.vmap(play_step_fn)(x),
                (starting_states, last_action_means, preferences, keys),
                length=ppo_configs.rollout_length,
            )
            
            final_states, final_action, _, __ = final_carry

            return final_states, final_action, transitions
        

        self._rollout_fn = rollout_fn


        def critic_loss_fn(
            critic_params: Params,
            transitions: PPOTransition,
        ) -> float:

            estimated_v = critic_network.apply(critic_params, transitions.obs, transitions.preferences)
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

                action_mean, _ = policy_network.apply(policy_params, transitions.obs, transitions.preferences)
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

        fake_obs = jnp.zeros(shape=(self._observation_size,))
        fake_preference = self._sample_fn(key)

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
            current_std=self.configs.initial_std,
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
            length=self.configs.critic_epochs,
        )

        (policy_params, policy_opt_state), _ = jax.lax.scan(
            lambda x, _: partial(self.train_policy, transitions=transitions)(x),
            (training_state.policy_params, training_state.policy_opt_state),
            length=self.configs.policy_epochs,
        )

        current_std = training_state.current_std - self.configs.std_decay_rate
        
        training_state = training_state.replace(
            policy_params=policy_params,
            critic_params=critic_params,
            policy_opt_state=policy_opt_state,
            critic_opt_state=critic_opt_state,
            current_std=jnp.clip(current_std, min=self.configs.min_std),
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
        ) -> Tuple[Tuple[Params, optax.OptState], Any]:
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



    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def train_critic(
        self,
        carry: Tuple[Params, optax.OptState],
        transitions: PPOTransition,
    ) -> Tuple[Tuple[Params, optax.OptState], Any]:
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
    

    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def train(
        self,
        starting_states: EnvState,
        last_action_means: jnp.ndarray,
        training_state: PPOTrainingState,
        key: RNGKey,
    ) -> Tuple[PPOTrainingState, EnvState]:
        """
        Perform one iteration of PPO update
        
        """

        policy_params = training_state.policy_params
        critic_params = training_state.critic_params
        current_std = training_state.current_std

        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, num=self.configs.vec_env)
        preferences = jax.vmap(self._sample_fn)(subkeys) # batch x d

        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, num=self.configs.vec_env)
        (
            states, last_actions, transitions
            ) = self._rollout_fn(policy_params, starting_states, last_action_means, preferences, subkeys, current_std)

        last_obs = self._build_obs(states.obs, last_actions)
        final_v_value = self._critic_network.apply(critic_params, last_obs, preferences)
        v_values = self.calculate_v(critic_params, transitions.obs, preferences)

        td_lambda_returns, weights = self.calculate_td_lambda_returns(
            final_v_value,
            v_values, 
            jnp.sum(transitions.rewards * preferences, axis=-1, keepdims=True), # rollout x batch x 1
            jnp.clip(1 - transitions.truncations - transitions.dones, min=0.0),
            ) # rollout x parallelize

        # mean_change = jnp.average(td_lambda_returns - v_values, axis=0, weights=weights, keepdims=True) # 1 x parallelize
        # td_lambda_returns = td_lambda_returns + (1 - weights) * mean_change # corrected td_lambda_returns

        gaes = td_lambda_returns - v_values
        gaes = self.process_gaes(gaes)

        transitions = transitions.replace(
            td_lambda_returns=td_lambda_returns,
            gaes=gaes,
            weights=weights,
            )
        
        key, subkey = jax.random.split(key)

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

        return (states, last_actions, new_training_state, key), transitions



    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def calculate_td_lambda_returns(
        self,
        final_v_value: jnp.ndarray,
        v_values: jnp.ndarray, 
        rewards: jnp.ndarray,
        masks: jnp.ndarray,
    ) -> jnp.ndarray:
        
        discount = self.configs.discount
        td_lambda_discount = self.configs.td_lambda_discount


        def scan_calculate_td_lambda(
            carry: jnp.ndarray, 
            data: Tuple[jnp.ndarray, jnp.ndarray],
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            
            (last_td_lambda_value, last_value, last_weight) = carry
            reward, v_value, mask = data
            current_td_lambda_value = reward + mask * discount * (
                    (1 - td_lambda_discount) * last_value + td_lambda_discount * last_td_lambda_value
                )
            weight = discount * td_lambda_discount * (last_weight - 1) * mask + 1

            return (current_td_lambda_value, v_value, weight), (current_td_lambda_value, weight)
            
        _, (td_lambda_values, weights) = jax.lax.scan(
            jax.vmap(scan_calculate_td_lambda),
            (final_v_value, final_v_value, jnp.zeros_like(final_v_value)),
            (rewards, v_values, masks),
            reverse=True,
        ) # length x batch x 1

        return td_lambda_values, weights


