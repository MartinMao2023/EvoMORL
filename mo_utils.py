from buffer import PPOTransition
from typing import Tuple, Any, Callable
from custom_types import RNGKey, Params, EnvState, Env
import jax
import jax.numpy as jnp
import math




def sample_task(key: RNGKey) -> jnp.ndarray:
    dim = 4
    preference_vector = jax.random.normal(key, (dim,))
    preference_vector = jnp.abs(preference_vector) / jnp.sqrt(1e-6 + jnp.sum(preference_vector**2))

    return preference_vector


def mo_ppo_exploraive_rollout(
    policy_fn: Callable,
    env_fn: Callable,
    rollout_length: int,
) -> Callable:

    
    @jax.jit
    def explorative_rollout_fn(
        policy_params: Params,
        starting_states: EnvState,
        last_action_means: jnp.ndarray,
        preferences: jnp.ndarray,
        keys: RNGKey,
        ) -> PPOTransition:

        def play_step_fn(
            carry: Tuple[EnvState, jnp.ndarray, int, RNGKey],
            ) -> Tuple[Tuple, PPOTransition]:
            
            state, last_action_mean, preference, key = carry
            key, subkey = jax.random.split(key)
            obs = jnp.concatenate([state.obs, last_action_mean], axis=-1)
            # obs = state.obs

            action_mean, action_std = policy_fn(policy_params, obs, preference)
            candidate_action_noise = action_std * jax.random.normal(subkey, action_mean.shape)
            action = jnp.clip(action_mean + candidate_action_noise, -1.0, 1.0)
            action_noise = action - action_mean
            action_log_std = jnp.log(action_std + 1e-6)

            next_state = env_fn(state, action)
            rewards = jnp.array([
                state.metrics["reward_forward"] + 1, 
                state.metrics["reward_ctrl"] + 0.25, 
                state.pipeline_state.x.pos[0, 2],
                1 - 2.5*jnp.mean(jnp.square(action_mean - last_action_mean)) # zero'th order smoothness
                ])

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
            length=rollout_length,
        )
        
        final_states, final_action, _, __ = final_carry

        return final_states, final_action, transitions
    
    return explorative_rollout_fn



def build_scoring_fn(env: Env, policy_fn: Callable, key: RNGKey, num: int, rollout_length: int=256) -> Callable:

    key1 , key2, key3 = jax.random.split(key, num=3)
    keys1 = jax.random.split(key1, num)
    keys2 = jax.random.split(key2, num)
    keys3 = jax.random.split(key3, num)

    starting_states = jax.vmap(env.reset)(keys1)
    preferences = jax.vmap(sample_task)(keys2) # goal conditioned version
    initial_action_mean = jnp.zeros((num, env.action_size))

    def scoring_fn(
        policy_params: Params,
        ) -> PPOTransition:

        def play_step_fn(
            carry: Tuple[EnvState, jnp.ndarray, int, RNGKey],
            ) -> Tuple[Tuple, PPOTransition]:
            
            state, last_action_mean, preference, key = carry
            key, subkey = jax.random.split(key)
            obs = jnp.concatenate([state.obs, last_action_mean], axis=-1)

            action_mean, action_std = policy_fn(policy_params, obs, preference)
            candidate_action_noise = action_std * jax.random.normal(subkey, action_mean.shape)
            action = jnp.clip(action_mean + candidate_action_noise, -1.0, 1.0)

            next_state = env.step(state, action)
            rewards = jnp.array([
                state.metrics["reward_forward"] + 1, 
                state.metrics["reward_ctrl"] + 0.25, 
                state.pipeline_state.x.pos[0, 2],
                1 - 2.5*jnp.mean(jnp.square(action_mean - last_action_mean))
                ])

            return (next_state, action_mean, preference, key), (rewards * preference, next_state.done)
        

        final_carry, (rewards, dones) = jax.lax.scan(
            lambda x, _: jax.vmap(play_step_fn)(x),
            (starting_states, initial_action_mean, preferences, keys3),
            length=rollout_length,
        )

        rewards = jnp.sum(rewards, axis=-1) # rollout x batch
        masks = jnp.clip(1 - jnp.cumsum(dones, axis=0), min=0.0)

        return jnp.mean(jnp.sum(rewards * masks, axis=1))

    return scoring_fn



def calculate_td_lambda_returns(
    final_v_value: jnp.ndarray,
    v_values: jnp.ndarray, 
    rewards: jnp.ndarray,
    masks: jnp.ndarray,
    discount: float, 
    td_lambda_discount: float,
    preferences: jnp.ndarray,
) -> jnp.ndarray:

    def scan_calculate_td_lambda(
        carry: jnp.ndarray, 
        data: Tuple[jnp.ndarray, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        
        (last_td_lambda_value, last_value, last_weight, preference) = carry
        reward, v_value, mask = data
        reward = jnp.sum(reward * preference, axis=-1)
        current_td_lambda_value = reward + mask * discount * (
                (1 - td_lambda_discount) * last_value + td_lambda_discount * last_td_lambda_value
            )
        weight = discount * td_lambda_discount * (last_weight - 1) * mask + 1

        return (current_td_lambda_value, v_value, weight, preference), (current_td_lambda_value, weight)
        
    _, (td_lambda_values, weights) = jax.lax.scan(
        jax.vmap(scan_calculate_td_lambda),
        (final_v_value, final_v_value, jnp.zeros_like(final_v_value), preferences),
        (rewards, v_values, masks),
        reverse=True,
    ) # length x batch x d

    return td_lambda_values, weights

