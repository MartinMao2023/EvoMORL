from buffer import PPOTransition
from typing import Tuple, Any, Callable
from custom_types import RNGKey, Params, EnvState
import jax
import jax.numpy as jnp
import math




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
        keys: RNGKey,
        ) -> PPOTransition:

        def play_step_fn(
            carry: Tuple[EnvState, jnp.ndarray, int, RNGKey],
            ) -> Tuple[Tuple, PPOTransition]:
            
            state, last_action_mean, key = carry
            key, subkey = jax.random.split(key)
            obs = jnp.concatenate([state.obs, last_action_mean], axis=-1)
            # obs = state.obs

            action_mean, action_std = policy_fn(policy_params, obs)
            candidate_action_noise = action_std * jax.random.normal(subkey, action_mean.shape)
            action = jnp.clip(action_mean + candidate_action_noise, -1.0, 1.0)
            action_noise = action - action_mean
            action_log_std = jnp.log(action_std + 1e-6)

            next_state = env_fn(state, action)
            # rewards=jnp.ones((1, )) * state.reward
            rewards = jnp.array([
                state.metrics["reward_forward"] + 1, 
                state.metrics["reward_ctrl"] + 0.25, 
                # 0.2 * jnp.mean(action_log_std) + 0.6, 
                state.pipeline_state.x.pos[0, 2],
                1 - 2.5*jnp.mean(jnp.square(action_mean - last_action_mean)) # zero'th order smoothness
                ])

            transition = PPOTransition(
                obs=obs,
                actions=action,
                action_noises=action_noise,
                action_log_std=action_log_std,
                # rewards=jnp.clip(state.reward - state.metrics["forward_reward"] + 3.0, min=0.0),
                rewards=rewards,
                td_lambda_returns=jnp.zeros((1,)),
                gaes=jnp.zeros((1,)),
                dones=next_state.done,
                # truncations=jnp.where(step_num < truncate_length, 0.0, 1.0),
                truncations=0.0,
                weights=jnp.zeros((1,)),
                )

            return (next_state, action_mean, key), transition

        final_carry, transitions = jax.lax.scan(
            lambda x, _: jax.vmap(play_step_fn)(x),
            (starting_states, last_action_means, keys),
            length=rollout_length,
        )
        
        final_states, final_action, _ = final_carry

        return final_states, final_action, transitions
    
    return explorative_rollout_fn




def local_ppo_exploraive_rollout(
    policy_fn: Callable,
    env_fn: Callable,
    rollout_length: int,
    initial_state: EnvState,
) -> Callable:
    
    # dt = 0.05 # for Ant
    # l = 0.5 # action length scale
    # discount_scale = 1 - dt / l
    # posterior_var = (1 - jnp.exp(-2 * dt/l)

    
    @jax.jit
    def explorative_rollout_fn(
        policy_params: Params,
        starting_states: EnvState,
        # last_action_mean: jnp.ndarray,
        steps: jnp.array,
        keys: RNGKey,
        ) -> PPOTransition:

        def play_step_fn(
            carry: Tuple[EnvState, jnp.ndarray, int, RNGKey],
            ) -> Tuple[Tuple, PPOTransition]:
            
            state, step, key = carry
            key, subkey = jax.random.split(key)
            obs = state.obs


            action_mean, action_log_std = policy_fn(policy_params, obs)
            candidate_action_noise = jnp.exp(action_log_std) * jax.random.normal(subkey, action_mean.shape)
            action = jnp.clip(action_mean + candidate_action_noise, -1.0, 1.0)
            action_noise = action - action_mean

            next_state = env_fn(state, action)
            rewards=jnp.ones((1, )) * state.reward
            # rewards = jnp.array([
            #     state.metrics["reward_forward"] + 1, 
            #     state.metrics["reward_ctrl"] + 0.25, 
            #     0.2 * jnp.mean(action_log_std) + 0.6, 
            #     # 1 - 0.5*jnp.mean(jnp.square(action_mean - discount_scale * last_action_mean))/posterior_var, 
            #     1 - 2.5*jnp.mean(jnp.square(action_mean - last_action_mean)) # zero'th order smoothness
            #     ])
            
            # key, subkey = jax.random.split(key)
            # p = jax.random.uniform(subkey, dtype=jnp.float32)
            # move_on = jnp.where(p < jnp.exp(((step/64 - 1)**2 - (step/64 - 63/64)**2)*0.5), 1.0, 0.0)
            # step_count = move_on * (step + 1) * (1 - next_state.done)

            move_on = jnp.where(step < 128, 1.0, 0.0)
            step_count = move_on * (step + 1) * (1 - next_state.done)


            state = jax.lax.cond(
                move_on,
                lambda x: next_state,
                lambda x: x,
                initial_state,
            )
            
            # state = jnp.where(move_on, next_state, initial_state)
            
            # prior std = 0.5

            transition = PPOTransition(
                obs=obs,
                actions=action,
                action_noises=action_noise,
                action_log_std=action_log_std,
                # rewards=jnp.clip(state.reward - state.metrics["forward_reward"] + 3.0, min=0.0),
                rewards=rewards,
                td_lambda_returns=jnp.zeros((1,)),
                gaes=jnp.zeros((1,)),
                dones=next_state.done,
                # truncations=jnp.where(step_num < truncate_length, 0.0, 1.0),
                truncations=1 - move_on,
                weights=jnp.zeros((1,)),
                )

            return (next_state, step_count, key), transition

        final_carry, transitions = jax.lax.scan(
            lambda x, _: jax.vmap(play_step_fn)(x),
            (starting_states, steps, keys),
            length=rollout_length,
        )
        
        final_states, step_count, _ = final_carry

        return final_states, step_count, transitions
    

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
    ) # length x batch x d

    return td_lambda_values, weights
        



def shuffle_transitions(key: RNGKey, transitions: PPOTransition) -> PPOTransition:
    flattened_transitions = transitions.flatten()
    num_transitions = flattened_transitions.shape[0]
    index = jax.random.permutation(key, num_transitions)
    transitions = transitions.__class__.from_flatten(flattened_transitions[index], transitions)
    
    return transitions

