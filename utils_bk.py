from buffer import PPOTransition
from typing import Tuple, Any, Callable
from custom_types import RNGKey, Params, EnvState
import jax
import jax.numpy as jnp
import math



def build_ppo_rollout(
    policy_fn: Callable,
    env_fn: Callable,
    rollout_length: int,
    noise_scale: float,
) -> Callable:
    
    @jax.jit
    def explorative_rollout_fn(
        policy_params: Params,
        starting_states: EnvState,
        keys: RNGKey,
        ) -> PPOTransition:

        def play_step_fn(
            carry: Tuple[EnvState, jnp.ndarray, int, RNGKey],
            ) -> Tuple[Tuple, PPOTransition]:
            
            state, key = carry
            key, subkey = jax.random.split(key)

            action_mean = policy_fn(policy_params, state.obs)
            candidate_action_noise = noise_scale * jax.random.normal(subkey, action_mean.shape)
            action = jnp.clip(action_mean + candidate_action_noise, -1.0, 1.0)
            action_noise = action - action_mean

            next_state = env_fn(state, action)
            rewards=jnp.ones((1, )) * state.reward

            transition = PPOTransition(
                obs=state.obs,
                actions=action,
                action_noises=action_noise,
                action_log_std=jnp.zeros_like(action),
                # rewards=jnp.clip(state.reward - state.metrics["forward_reward"] + 3.0, min=0.0),
                rewards=rewards,
                td_lambda_returns=jnp.zeros_like(rewards),
                gaes=jnp.zeros_like(rewards),
                dones=next_state.done,
                # truncations=jnp.where(step_num < truncate_length, 0.0, 1.0),
                truncations=0,
                weights=0.0,
                )

            return (next_state, key), transition

        final_carry, transitions = jax.lax.scan(
            lambda x, _: jax.vmap(play_step_fn)(x),
            (starting_states, keys),
            length=rollout_length,
        )
        
        final_states, _ = final_carry

        return final_states, transitions
    

    return explorative_rollout_fn






def calculate_td_lambda_returns(
    final_v_value: jnp.ndarray,
    v_values: jnp.ndarray, 
    rewards: jnp.ndarray,
    masks: jnp.ndarray,
    discount: float, 
    td_lambda_discount: float,
) -> jnp.ndarray:

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
    )

    return td_lambda_values, weights



def shuffle_transitions(key: RNGKey, transitions: PPOTransition) -> PPOTransition:
    flattened_transitions = transitions.flatten()
    num_transitions = flattened_transitions.shape[0]
    index = jax.random.permutation(key, num_transitions)
    transitions = transitions.__class__.from_flatten(flattened_transitions[index], transitions)
    
    return transitions





# def build_ppo_rollout(
#     policy_fn: Callable,
#     env_fn: Callable,
#     rollout_length: int,
# ) -> Callable:
    
#     @jax.jit
#     def explorative_rollout_fn(
#         policy_params: Params,
#         starting_states: EnvState,
#         keys: RNGKey,
#         ) -> PPOTransition:

#         def play_step_fn(
#             carry: Tuple[EnvState, jnp.ndarray, int, RNGKey],
#             ) -> Tuple[Tuple, PPOTransition]:
            
#             state, key = carry
#             key, subkey = jax.random.split(key)

#             action_mean, action_log_std = policy_fn(policy_params, state.obs)
#             candidate_action_noise = jnp.exp(action_log_std) * jax.random.normal(subkey, action_mean.shape)
#             action = jnp.clip(action_mean + candidate_action_noise, -1.0, 1.0)
#             action_noise = action - action_mean

#             next_state = env_fn(state, action)
#             rewards=jnp.ones((1, )) * state.reward

#             transition = PPOTransition(
#                 obs=state.obs,
#                 actions=action,
#                 action_noises=action_noise,
#                 action_log_std=action_log_std,
#                 # rewards=jnp.clip(state.reward - state.metrics["forward_reward"] + 3.0, min=0.0),
#                 rewards=rewards,
#                 td_lambda_returns=jnp.zeros_like(rewards),
#                 gaes=jnp.zeros_like(rewards),
#                 dones=next_state.done,
#                 # truncations=jnp.where(step_num < truncate_length, 0.0, 1.0),
#                 truncations=0,
#                 weights=0.0,
#                 )

#             return (next_state, key), transition

#         final_carry, transitions = jax.lax.scan(
#             lambda x, _: jax.vmap(play_step_fn)(x),
#             (starting_states, keys),
#             length=rollout_length,
#         )
        
#         final_states, _ = final_carry

#         return final_states, transitions
    

#     return explorative_rollout_fn


