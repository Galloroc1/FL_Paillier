from paillier import generate_paillier_keypair,toArray

import numpy as np
import datetime

import jax
import jax.numpy as jnp
import jax.config as config
config.update('jax_enable_x64', True)
from np_paillier.paillier import generate_paillier_keypair,toArray
p,q = generate_paillier_keypair()
data = np.random.random_sample((10,10))
data_jax = jnp.array(data,dtype=jnp.float64)
print(data_jax)
print(data)
st = datetime.datetime.now()
r = p.encrypt(data) + data_jax
print(datetime.datetime.now()-st)
r2 = q.decrypt(r)
print(data[1][1]+data[1][1])
print(r2[1][1])