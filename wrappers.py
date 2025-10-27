from typing import Callable, Dict, Optional, Tuple

# from brax.base import System
from brax.envs.base import Env, State, Wrapper
# from flax import struct
import jax
from jax import numpy as jp



class AutoResetWrapper(Wrapper):
  """Automatically resets Brax envs that are done."""

  def reset(self, rng: jax.Array) -> State:
    rng1, rng2 = jax.random.split(rng)
    state = self.env.reset(rng1)
    backup_state = self.env.reset(rng2)
    state.info['first_pipeline_state'] = backup_state.pipeline_state
    state.info['first_obs'] = backup_state.obs
    return state
  

  def refresh_backup_state(self, state: State, rng: jax.Array) -> State:
    backup_state = self.env.reset(rng)
    state.info['first_pipeline_state'] = backup_state.pipeline_state
    state.info['first_obs'] = backup_state.obs
    return state
  

  def step(self, state: State, action: jax.Array) -> State:
    if 'steps' in state.info:
      steps = state.info['steps']
      steps = jp.where(state.done, jp.zeros_like(steps), steps)
      state.info.update(steps=steps)
    state = state.replace(done=jp.zeros_like(state.done))
    state = self.env.step(state, action)

    def where_done(x, y):
      done = state.done
      if done.shape:
        done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
      return jp.where(done, x, y)

    pipeline_state = jax.tree.map(
        where_done, state.info['first_pipeline_state'], state.pipeline_state
    )
    obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)
    return state.replace(pipeline_state=pipeline_state, obs=obs)
  