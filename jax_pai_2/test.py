from paillier import generate_paillier_keypair
import jax
import numpy as np
import jax.numpy as jnp
data_np = np.random.random_sample((2,2))
data = jnp.asarray(data_np)
p,q = generate_paillier_keypair()
d2 = p.encrypt(data)
print(d2)