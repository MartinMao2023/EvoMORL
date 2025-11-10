from flax.struct import dataclass

from functools import partial
from typing import Any, Tuple, Callable

import flax.linen as nn
import jax
import optax
from jax import numpy as jnp


from buffer import PPOTransition
from algorithms import PPOConfigs
from networks import QModuleTC
from custom_types import Params, RNGKey, Env, EnvState
from flax.struct import PyTreeNode
from res_ppo import ResPPO, ResPPOTrainingState
# from mo_utils import sample_task
# from losses import bce_loss



@dataclass
class ILConfigs:
    stage_steps: int = 16
    stage_iter: int = 8
    popsize: int = 64
    policy_epochs: int = 4
    critic_epochs: int = 4
    mini_batch_size: int = 1024
    policy_learnng_rate: float = 5e-4 # imitation learning
    critic_learning_rate: float = 5e-4 # imitation learning

    initial_std: jnp.ndarray = 0.15
    std_decay_rate: float = 5e-5
    min_std: jnp.ndarray = 0.1




class ILTrainingState(PyTreeNode):
    """Contains training state for the learner."""

    critic_params: Params
    policy_params: Params

    critic_opt_state: optax.OptState
    policy_opt_state: optax.OptState

    current_std: jnp.ndarray

    




class IL_emitter:
    def __init__(
        self,
        env: Env,
        policy_network: nn.Module,
        critic_network: nn.Module,
        offspring_network: nn.Module,
        baseline_network: nn.Module,
        il_configs: ILConfigs,
        ppo_configs: PPOConfigs,
        sample_fn: Callable,
        include_last_action_in_obs: bool = True,
        ):

        self._env = env
        self._sample_fn = sample_fn
        self._configs = il_configs
        self._ppo_configs = ppo_configs

        self._pg_steps = int(self._configs.stage_steps * self._configs.stage_iter)

        self._include_last_action_in_obs = include_last_action_in_obs
        self._res_ppo = ResPPO(
            env,
            offspring_network,
            baseline_network,
            ppo_configs,
            include_last_action_in_obs,
        )

        self._policy_network = policy_network
        self._critic_network = critic_network
        # self._offspring_network = offspring_network

        
        if self._include_last_action_in_obs:
            self._observation_size = self._env.observation_size + self._env.action_size
        else:
            self._observation_size = self._env.observation_size

        self._policy_optimizer = optax.adam(
            learning_rate=il_configs.policy_learnng_rate,
        )
        self._critic_optimizer = optax.adam(
            learning_rate=il_configs.critic_learning_rate,
        )


        def critic_loss_fn(
            critic_params: Params,
            transitions: PPOTransition,
        ) -> float:

            estimated_v = critic_network.apply(critic_params, transitions.obs, transitions.preferences)
            weights = 1 / (100 - 99*transitions.weights)

            return jnp.mean(jnp.square(estimated_v - transitions.td_lambda_returns) * weights)
        
        self._critic_loss_fn = critic_loss_fn


        def policy_loss_fn(
            policy_params: Params,
            transitions: PPOTransition,
        ) -> float:
            
            action_mean, _ = policy_network.apply(policy_params, transitions.obs, transitions.preferences)

            return jnp.mean(jnp.square(action_mean - transitions.actions + transitions.action_noises) * transitions.weights)

    
        self._policy_loss_fn = policy_loss_fn



    def init(
        self, 
        key: RNGKey,
    ) -> ILTrainingState:

        fake_obs = jnp.zeros(shape=(self._observation_size,))
        fake_preference = self._sample_fn(key)

        key, subkey = jax.random.split(key)
        policy_params = self._policy_network.init(subkey, fake_obs, fake_preference)
        policy_opt_state = self._policy_optimizer.init(policy_params)

        key, subkey = jax.random.split(key)
        critic_params = self._critic_network.init(subkey, fake_obs, fake_preference)
        critic_opt_state = self._critic_optimizer.init(critic_params)
        
        training_state = ILTrainingState(
            critic_params=critic_params,
            policy_params=policy_params,
            critic_opt_state=critic_opt_state,
            policy_opt_state=policy_opt_state,
            current_std=self._configs.initial_std,
            )
        
        return training_state
    


    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def state_update(
        self, 
        training_state: ILTrainingState, 
        transitions: PPOTransition, 
    ) -> ILTrainingState:
        
        transitions = jax.tree.map(
            lambda x: jnp.reshape(
                jnp.swapaxes(x, 0, 2),
                (
                    -1,
                    self._configs.mini_batch_size,
                    *x.shape[3:],
                ),
            ),
            transitions)

        (critic_params, critic_opt_state), critic_learning_curve = jax.lax.scan(
            lambda x, _: partial(self.train_critic, transitions=transitions)(x),
            (training_state.critic_params, training_state.critic_opt_state),
            length=self._configs.critic_epochs,
        )

        (policy_params, policy_opt_state), policy_learning_curve = jax.lax.scan(
            lambda x, _: partial(self.train_policy, transitions=transitions)(x),
            (training_state.policy_params, training_state.policy_opt_state),
            length=self._configs.policy_epochs,
        )

        current_std = training_state.current_std - self._configs.std_decay_rate
        
        training_state = training_state.replace(
            policy_params=policy_params,
            critic_params=critic_params,
            policy_opt_state=policy_opt_state,
            critic_opt_state=critic_opt_state,
            current_std=jnp.clip(current_std, min=self._configs.min_std),
        )

        return training_state, policy_learning_curve, critic_learning_curve



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

            policy_error, policy_gradient = jax.value_and_grad(self._policy_loss_fn)(
                current_policy_params,
                transition_data,
                )

            policy_updates, new_policy_opt_state = self._policy_optimizer.update(
                policy_gradient, current_policy_opt_state)
            new_policy_params = optax.apply_updates(current_policy_params, policy_updates)
            
            return (new_policy_params, new_policy_opt_state), policy_error
        

        (final_policy_params, final_policy_opt_state), policy_errors = jax.lax.scan(
            scan_train_policy,
            (policy_params, policy_opt_state),
            transitions,
        )
        
        return (final_policy_params, final_policy_opt_state), jnp.mean(policy_errors)



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

            critic_error, critic_gradient = jax.value_and_grad(self._critic_loss_fn)(
                current_critic_params,
                transition_data,
                )

            critic_updates, new_critic_opt_state = self._critic_optimizer.update(
                critic_gradient, current_critic_opt_state)
            new_critic_params = optax.apply_updates(current_critic_params, critic_updates)
            
            return (new_critic_params, new_critic_opt_state), critic_error
        

        (final_critic_params, final_critic_opt_state), critic_errors = jax.lax.scan(
            scan_train_critic,
            (critic_params, critic_opt_state),
            transitions,
        )
        
        return (final_critic_params, final_critic_opt_state), jnp.mean(critic_errors)
    


    def emit(
        self, 
        training_state: ILTrainingState,
        key: RNGKey,
    ) -> Tuple[Params, jnp.ndarray]:

        keys = jax.random.split(key, num=self._configs.popsize)
        preferences = jax.vmap(self._sample_fn)(keys)

        policy_params = jax.vmap(
            self._compute_equivalent_params_with_preference, in_axes=(None, 0)
            )(training_state.policy_params, preferences)

        critic_params = jax.vmap(
            self._compute_equivalent_params_with_preference, in_axes=(None, 0)
            )(training_state.critic_params, preferences)

        return policy_params, critic_params, preferences



    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def mutate_pg(
        self,
        policies: Params,
        critics: Params, 
        starting_states: EnvState,
        starting_actions: jnp.ndarray,
        preferences: jnp.ndarray,
        key: RNGKey,
        current_std: jnp.ndarray,
    ) -> PPOTransition:

        (
            starting_states, 
            starting_actions,
            ) = jax.vmap(self.duplicate)(starting_states, starting_actions)

        subkey, key = jax.random.split(key)
        subkeys = jax.random.split(subkey, num=self._configs.popsize)
        keys = jax.random.split(key, num=self._configs.popsize)

        ppo_training_states = jax.vmap(partial(self._res_ppo.init, current_std=current_std))(
            policies,
            critics,
            preferences,
            subkeys,
        ) # popsize


        (
            states, 
            last_actions, 
            ppo_training_states,
            keys,
        ), transitions = jax.vmap(self._res_ppo.train)(
            starting_states,
            starting_actions,
            ppo_training_states,
            keys,
        )

        def scan_ppo(carry, data):
            (
                states, 
                last_actions, 
                ppo_training_states,
                keys,
                transitions,
            ) = carry

            (
                states, 
                last_actions, 
                ppo_training_states,
                keys,
            ), transitions = self._res_ppo.train(
                states,
                last_actions,
                ppo_training_states,
                keys,
            )

            new_carry = (
                states, 
                last_actions, 
                ppo_training_states,
                keys,
                transitions,
            )

            return new_carry, jnp.mean(transitions.td_lambda_returns)


        # for i in range(32):
        #     (
        #         states, 
        #         last_actions, 
        #         ppo_training_states,
        #         keys,
        #     ), transitions = jax.vmap(self._res_ppo.train)(
        #         states,
        #         last_actions,
        #         ppo_training_states,
        #         keys,
        #     )

        (
            states, 
            last_actions, 
            ppo_training_states,
            keys,
            transitions,
            ), learning_curves = jax.lax.scan(
                jax.vmap(scan_ppo),
                (states, last_actions, ppo_training_states, keys, transitions),
                length=128
            )
        

        
        # subkeys, keys = jax.vmap(jax.random.split)(keys)
        states, last_actions = jax.vmap(self.select_states)(
            states,
            last_actions,
            keys,
        )

        return (states, last_actions), transitions, learning_curves


    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def staged_ppo_training(
        self,
        ) -> Tuple[Tuple, RNGKey, ResPPOTrainingState]:

        return 



    @partial(
        jax.jit,
        static_argnames=("self",)
    )
    def train(
        self,
        training_state: ILTrainingState,
        starting_states: EnvState,
        starting_actions: jnp.ndarray,
        key: RNGKey,
    ) -> ILTrainingState:
        
        # subkey, key = jax.random.split(key)
        subkey = jax.random.PRNGKey(66)
        policies, critics, preferences = self.emit(training_state, subkey) # randomly sample policies with critics

        # pass these to staged ppo training, collect transitions
        subkey, key = jax.random.split(key)

        # subkeys = jax.random.split(subkey, num=self._configs.popsize)
        (states, last_actions), transitions, ppo_learning_curves = self.mutate_pg(
            policies, 
            critics, 
            starting_states, 
            starting_actions,
            preferences, 
            subkey,
            training_state.current_std,
            ) # (popsize x vec_env x ?), (popsize x batchsize x step_num)


        # distillate these (without reweighing) into the global policy
        new_training_state, policy_learning_curve, critic_learning_curve = self.state_update(training_state, transitions)
        learning_curves = (ppo_learning_curves, policy_learning_curve, critic_learning_curve)
        
        return (new_training_state, states, last_actions, key), learning_curves # rollout_length x popsize
    


    @partial(jax.jit, static_argnames=("self",))
    def _compute_equivalent_params_with_preference(
        self, universal_params: Params, preference: jnp.ndarray,
    ) -> Params:
        
        kernel = universal_params["params"]["Dense_0"]["kernel"]
        bias = universal_params["params"]["Dense_0"]["bias"]
        equivalent_kernel = kernel[: -preference.shape[0], :]
        equivalent_bias = bias + jnp.dot(preference, kernel[-preference.shape[0] :])

        universal_params["params"]["Dense_0"]["kernel"] = equivalent_kernel
        universal_params["params"]["Dense_0"]["bias"] = equivalent_bias
        return universal_params
    
        

    @partial(
        jax.jit,
        static_argnames=("self",)
    )
    def duplicate(
        self, 
        state: EnvState,
        action: jnp.ndarray,
        ) -> Tuple[EnvState, jnp.ndarray]:

        # states = jax.tree.map(
        #         lambda x: jnp.tile(x, self._ppo_configs.vec_env),
        #         state
        #     )
        
        states = jax.tree.map(
                lambda x: jnp.repeat(
                    x[None, ...], 
                    self._ppo_configs.vec_env, 
                    axis=0
                    ),
                state

                
            )
        actions = jnp.tile(action, (self._ppo_configs.vec_env, 1))


        return states, actions


    @partial(
        jax.jit,
        static_argnames=("self",)
    )
    def select_states(
        self, 
        states: EnvState,
        actions: jnp.ndarray,
        key: RNGKey,
        ) -> Tuple[EnvState, jnp.ndarray]:

        index = jax.random.choice(key, self._ppo_configs.vec_env)
        state = jax.tree.map(
                lambda x: x[index],
                states
            )
        action = actions[index]

        return state, action


