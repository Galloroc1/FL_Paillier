from paillier import generate_paillier_keypair, toArray

import numpy as np
import datetime
import jax
import jax.numpy as jnp
import jax.config as config
import pickle

config.update('jax_enable_x64', True)
from np_paillier.paillier import generate_paillier_keypair,toArray

p, q = generate_paillier_keypair()

# def en_data_add_jax_data(data, data_jax):
#     r = p.encrypt(data) + data_jax
#     return r
#
# def en_data_add_test_dot(data,data_jax):
#     data_en = p.encrypt(data)
#
#     # return r
#     # jnp_r = jax.vmap(jnp.add)(data_jax,r)
#
#
# def en_data_sub_mul_data(data, data_jax):
#     r = p.encrypt(data) * data_jax
#     return r
#
#
data = np.random.random_sample((100, 100))
# # data_jax = jnp.array(data, dtype=jnp.float64)
import dask.array as da
import mars.tensor as mt
from dask.distributed import Client,LocalCluster
# 最简单的方式，默认按照cpu核数创建worker数量
def train():
    r = toArray(p.encrypt(data))
    r = da.from_array(r,chunks=10)
    # r = mt.array(r,chunk_size=1024)
    st = datetime.datetime.now()
    r = da.dot(r,data.T).compute()
    # r = mt.dot(r,data.T).execute()
    print(r)
    # 100*100 56.14

    print(datetime.datetime.now()-st)

if __name__ == '__main__':

    c = Client()
    # 也可以指定参数
    c = Client(LocalCluster(n_workers=16, threads_per_worker=12, processes=True))
    train()



# r = en_data_add_jax_data(data, data_jax)
# r = toArray(en_data_add_jax_data(data,data_jax))
