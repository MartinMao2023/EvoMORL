from buffer import Transition
from typing import Tuple, Any, Callable
from custom_types import RNGKey, Params, EnvState
import jax
import jax.numpy as jnp



def bce_loss(
    y: jnp.ndarray, 
    target: jnp.ndarray
    ) -> float:
    target = 0.0005 + 0.999*target
    losses = -target * jnp.log(y + 1e-6) + \
        (target - 1) * jnp.log(1 + 1e-6 - y)
    
    return jnp.mean(losses)



def brpg_projection(
    main_grad: Params, 
    secondary_grad: Params
    ) -> Params:
    
    pass

