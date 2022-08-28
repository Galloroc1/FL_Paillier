from util import getprimeover
from key import PublicKey

SUPPORT_TYPE = ["float", "int", "numpy.ndarray"]
DEFAULT_KEYSIZE = 1024


def generate_paillier_keypair(n_length=DEFAULT_KEYSIZE)->PublicKey:
    """Return a new :class:`PaillierPublicKey` and :class:`PaillierPrivateKey`.

    Add the private key to *private_keyring* if given.

    Args:
      private_keyring (PaillierPrivateKeyring): a
        :class:`PaillierPrivateKeyring` on which to store the private
        key.
      n_length: key size in bits.

    Returns:
      tuple: The generated :class:`PaillierPublicKey` and
      :class:`PaillierPrivateKey`
    """
    p = q = n = None
    n_len = 0
    while n_len != n_length:
        p = getprimeover(n_length // 2)
        q = p
        while q == p:
            q = getprimeover(n_length // 2)
        n = p * q
        n_len = n.bit_length()
    public_key = PublicKey(n)
    return public_key
   # private_key = PaillierPrivateKey(public_key, p, q)

    # return public_key, private_key

