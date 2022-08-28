from paillier import generate_paillier_keypair, toArray

import numpy as np
import datetime
import jax
import jax.numpy as jnp
import jax.config as config

config.update('jax_enable_x64', True)
from np_paillier.paillier import generate_paillier_keypair, toArray

p, q = generate_paillier_keypair()


def en_data_add_jax_data(data, data_jax):
    r = p.encrypt(data) + data_jax
    return r

def en_data_add_test_dot(data,data_jax):
    data_en = p.encrypt(data)
    st = datetime.datetime.now()
    r = data_en.dot(data,is_pool=False)
    print(datetime.datetime.now()-st)
    print(r.shape)
    # return r
    # jnp_r = jax.vmap(jnp.add)(data_jax,r)


def en_data_sub_mul_data(data, data_jax):
    r = p.encrypt(data) * data_jax
    return r


data = np.random.random_sample((100, 100))
data_jax = jnp.array(data, dtype=jnp.float64)
r = en_data_add_test_dot(data,data_jax)
# r = en_data_add_jax_data(data, data_jax)
# r = toArray(en_data_add_jax_data(data,data_jax))
print(r)