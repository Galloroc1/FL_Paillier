import logging
import random
import numpy as np
import datetime
from functools import reduce
import multiprocessing as mp

try:
    from collections.abc import Mapping
except ImportError:
    Mapping = dict

from .encoding import EncodedNumber
from .util import invert, powmod, getprimeover, isqrt,trans_nptype
from functools import wraps
SUPPORT_TYPE = ["float", "int", "numpy.ndarray"]
DEFAULT_KEYSIZE = 1024

def toEncryptedNumber(arr):
    data = np.frompyfunc(lambda x: (x.ciphertext(False), x.exponent), 1, 2)(arr)
    public_key = arr.flatten()[0].public_key
    return EncryptedNumber(public_key, data[0], data[1])

class Paillier:
    NAME = "Paillier"

    def __init__(self):
        pass

    def createKey(self):
        """"
        create public_key and private_key
        :return public_key and private_key
        """
        self.public_key, self.private_key = self.generate_paillier_keypair()
        return self.public_key, self.private_key

    def encrypt(self, arr, is_pool=False):
        return self.public_key.encrypt(arr, is_pool=is_pool)

    def decrypt(self, arr, is_pool=False):
        """
        :param arr: np.ndarray or EncryptedNumber()
        :return: np.ndarray
        """
        return self.private_key.decrypt(arr, is_pool=is_pool)

    def generate_paillier_keypair(self, n_length=DEFAULT_KEYSIZE):
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
        public_key = PaillierPublicKey(n)
        private_key = PaillierPrivateKey(public_key, p, q)

        return public_key, private_key


class PaillierPublicKey(object):
    """Contains a public key and associated encryption methods.

    Args:

      n (int): the modulus of the public key - see Paillier's paper.

    Attributes:
      g (int): part of the public key - see Paillier's paper.
      n (int): part of the public key - see Paillier's paper.
      nsquare (int): :attr:`n` ** 2, stored for frequent use.
      max_int (int): Maximum int that may safely be stored. This can be
        increased, if you are happy to redefine "safely" and lower the
        chance of detecting an integer overflow.
    """
    def __init__(self, n):
        self.g = n + 1
        self.n = n
        self.nsquare = n * n
        self.max_int = n // 3 - 1

    def __repr__(self):
        publicKeyHash = hex(hash(self))[2:]
        return "<PaillierPublicKey {}>".format(publicKeyHash[:10])

    def __eq__(self, other):
        return self.n == other.n

    def __hash__(self):
        return hash(self.n)

    def get_nude(self, x):
        if x < self.n and self.n - self.max_int <= x:
            neg_plaintext = self.n - x  # = abs(plaintext - nsquare)
            neg_ciphertext = (self.n * neg_plaintext + 1) % self.nsquare
            nude_ciphertext = invert(neg_ciphertext, self.nsquare)
        else:
            # we chose g = n + 1, so that we can exploit the fact that
            # (n+1)^plaintext = n*plaintext + 1 mod n^2
            nude_ciphertext = (self.n * x + 1) % self.nsquare
        return nude_ciphertext

    def raw_encrypt(self, plaintext, r_value=None):
        """Paillier encryption of a positive integer plaintext < :attr:`n`.

        You probably should be using :meth:`encrypt` instead, because it
        handles positive and negative ints and floats.

        Args:
          plaintext (int): a positive integer < :attr:`n` to be Paillier
            encrypted. Typically this is an encoding of the actual
            number you want to encrypt.
          r_value (int): obfuscator for the ciphertext; by default (i.e.
            r_value is None), a random value is used.

        Returns:
          int: Paillier encryption of plaintext.

        Raises:
          TypeError: if plaintext is not an int.
        """
        nude_ciphertext = np.mod(self.n * plaintext + 1, self.nsquare)
        r = r_value or self.get_random_lt_n()
        obfuscator = powmod(r, self.n, self.nsquare)
        return np.mod(nude_ciphertext * obfuscator, self.nsquare)

    def get_random_lt_n(self):
        """Return a cryptographically random number less than :attr:`n`"""
        return random.SystemRandom().randrange(1, self.n)

    def encrypt_one(self, value, precision=None, r_value=None):

        if isinstance(value, EncodedNumber):
            encoding = value
        else:
            encoding = EncodedNumber.encode(self, value, precision)
        return self.encrypt_encoded(encoding, r_value)

    def raw_exponent_ciphertext(self, x, ):
        tw = self.encrypt_one(x)
        return [tw.exponent, tw.ciphertext(False)]

    def encrypt(self, arr, is_pool=False):
        """Encode and Paillier encrypt a real number *value*.

        Args:
          value: an int or float to be encrypted.
            If int, it must satisfy abs(*value*) < :attr:`n`/3.
            If float, it must satisfy abs(*value* / *precision*) <<
            :attr:`n`/3
            (i.e. if a float is near the limit then detectable
            overflow may still occur)
          precision (float): Passed to :meth:`EncodedNumber.encode`.
            If *value* is a float then *precision* is the maximum
            **absolute** error allowed when encoding *value*. Defaults
            to encoding *value* exactly.
          r_value (int): obfuscator for the ciphertext; by default (i.e.
            if *r_value* is None), a random value is used.

        Returns:
          EncryptedNumber: An encryption of *value*.

        Raises:
          ValueError: if *value* is out of range or *precision* is so
            high that *value* is rounded to zero.
        """
        if isinstance(arr, np.ndarray) or isinstance(arr, float) or isinstance(arr, int):
            pass
        else:
            raise TypeError(f"check your data type! {type(arr)} is not support type! we only support :{SUPPORT_TYPE}")

        if isinstance(arr, np.ndarray):
            if len(arr.shape)==2:
                if is_pool:
                    # 矩阵可选pool 标量和向量不可选pool
                    if arr.shape[0] >= 64:
                        pool = mp.Pool(processes=mp.cpu_count()-1)
                        data = np.array(pool.map(self.raw_exponent_ciphertext, arr))
                        pool.close()
                        pool.join()
                        results = EncryptedNumber(self, data[:, 1, :], data[:, 0, :])
                        return results
                    else:
                        return self.encrypt_one(arr)
            return self.encrypt_one(arr)
        else:
            return self.encrypt_one(arr)

    def encrypt_encoded(self, encoding, r_value):
        """Paillier encrypt an encoded value.

        Args:
          encoding: The EncodedNumber instance.
          r_value (int): obfuscator for the ciphertext; by default (i.e.
            if *r_value* is None), a random value is used.

        Returns:
          EncryptedNumber: An encryption of *value*.
        """
        obfuscator = r_value or 1
        ciphertext = self.raw_encrypt(encoding.encoding, r_value=obfuscator)
        encrypted_number = EncryptedNumber(self, ciphertext, encoding.exponent)
        if r_value is None:
            encrypted_number.obfuscate()
        return encrypted_number


class PaillierPrivateKey(object):
    """Contains a private key and associated decryption method.

    Args:
      public_key (:class:`PaillierPublicKey`): The corresponding public
        key.
      p (int): private secret - see Paillier's paper.
      q (int): private secret - see Paillier's paper.

    Attributes:
      public_key (PaillierPublicKey): The corresponding public
        key.
      p (int): private secret - see Paillier's paper.
      q (int): private secret - see Paillier's paper.
      psquare (int): p^2
      qsquare (int): q^2
      p_inverse (int): p^-1 mod q
      hp (int): h(p) - see Paillier's paper.
      hq (int): h(q) - see Paillier's paper.
    """
    def __init__(self, public_key, p, q):
        if not p * q == public_key.n:
            raise ValueError('given public key does not match the given p and q.')
        if p == q:
            # check that p and q are different, otherwise we can't compute p^-1 mod q
            raise ValueError('p and q have to be different')
        self.public_key = public_key
        if q < p:  # ensure that p < q.
            self.p = q
            self.q = p
        else:
            self.p = p
            self.q = q
        self.psquare = self.p * self.p

        self.qsquare = self.q * self.q
        self.p_inverse = invert(self.p, self.q)
        self.hp = self.h_function(self.p, self.psquare)
        self.hq = self.h_function(self.q, self.qsquare)

    # @staticmethod
    def from_totient(public_key, totient):
        """given the totient, one can factorize the modulus

        The totient is defined as totient = (p - 1) * (q - 1),
        and the modulus is defined as modulus = p * q

        Args:
          public_key (PaillierPublicKey): The corresponding public
            key
          totient (int): the totient of the modulus

        Returns:
          the :class:`PaillierPrivateKey` that corresponds to the inputs

        Raises:
          ValueError: if the given totient is not the totient of the modulus
            of the given public key
        """
        p_plus_q = public_key.n - totient + 1
        p_minus_q = isqrt(p_plus_q * p_plus_q - public_key.n * 4)
        q = (p_plus_q - p_minus_q) // 2
        p = p_plus_q - q
        if not p * q == public_key.n:
            raise ValueError('given public key and totient do not match.')
        return PaillierPrivateKey(public_key, p, q)

    def __repr__(self):
        pub_repr = repr(self.public_key)
        return "<PaillierPrivateKey for {}>".format(pub_repr)

    def decrypt_one(self, encrypted_number):
        encoded = self.decrypt_encoded(encrypted_number)
        return encoded.decode()

    def decrypt(self, arr, is_pool=False):
        """Return the decrypted & decoded plaintext of *encrypted_number*.

        Uses the default :class:`EncodedNumber`, if using an alternative encoding
        scheme, use :meth:`decrypt_encoded` or :meth:`raw_decrypt` instead.

        Args:
          encrypted_number (EncryptedNumber): an
            :class:`EncryptedNumber` with a public key that matches this
            private key.

        Returns:
          the int or float that `EncryptedNumber` was holding. N.B. if
            the number returned is an integer, it will not be of type
            float.

        Raises:
          TypeError: If *encrypted_number* is not an
            :class:`EncryptedNumber`.
          ValueError: If *encrypted_number* was encrypted against a
            different key.
        """
        if isinstance(arr, EncryptedNumber):
            if is_pool:
                arr = arr.toArray()
                if arr.shape[0] >= 64:
                    pool = mp.Pool(processes=mp.cpu_count()-1)
                    result = np.array(pool.map(self.decrypt_one, arr))
                    pool.close()
                    pool.join()
                    return trans_nptype(result)
                else:
                    return trans_nptype(self.decrypt_one(arr))
            else:
                return self.decrypt_one(arr)
        elif isinstance(arr,np.ndarray):
            result = self.decrypt_one(arr)
            return trans_nptype(result)
        else:
            raise TypeError("not support type,need {np.ndarray,EncryptedNumber}")


    def decrypt_encoded(self, encrypted_number, Encoding=None):
        """Return the :class:`EncodedNumber` decrypted from *encrypted_number*.

        Args:
          encrypted_number (EncryptedNumber): an
            :class:`EncryptedNumber` with a public key that matches this
            private key.
          Encoding (class): A class to use instead of :class:`EncodedNumber`, the
            encoding used for the *encrypted_number* - used to support alternative
            encodings.

        Returns:
          :class:`EncodedNumber`: The decrypted plaintext.

        Raises:
          TypeError: If *encrypted_number* is not an
            :class:`EncryptedNumber`.
          ValueError: If *encrypted_number* was encrypted against a
            different key.
        """
        if not isinstance(encrypted_number, EncryptedNumber) and not isinstance(encrypted_number, np.ndarray):
            raise TypeError('Expected encrypted_number to be an EncryptedNumber'
                            ' not: %s' % type(encrypted_number))

        if isinstance(encrypted_number, np.ndarray):
            encrypted_number = self.to_encrypted_number(encrypted_number)

        if self.public_key != encrypted_number.public_key:
            raise ValueError('encrypted_number was encrypted against a '
                             'different key!')

        if Encoding is None:
            Encoding = EncodedNumber

        encoded = self.raw_decrypt(encrypted_number.ciphertext(be_secure=False))
        encoded = np.frompyfunc(int, 1, 1)(encoded)
        return Encoding(self.public_key, encoded, encrypted_number.exponent)

    def to_encrypted_number(self, array):
        data = np.frompyfunc(lambda x: (x.ciphertext(False), x.exponent), 1, 2)(array)
        return EncryptedNumber(self.public_key, data[0], data[1])

    def raw_decrypt(self, ciphertext):
        """Decrypt raw ciphertext and return raw plaintext.

        Args:
          ciphertext (int): (usually from :meth:`EncryptedNumber.ciphertext()`)
            that is to be Paillier decrypted.

        Returns:
          int: Paillier decryption of ciphertext. This is a positive
          integer < :attr:`public_key.n`.

        Raises:
          TypeError: if ciphertext is not an int.
        """
        f = np.frompyfunc(powmod, 3, 1)
        decrypt_to_p = np.mod(np.floor_divide(f(ciphertext, self.p - 1, self.psquare) - 1, self.p) * self.hp, self.p)
        decrypt_to_q = np.mod(np.floor_divide(f(ciphertext, self.q - 1, self.qsquare) - 1, self.q) * self.hq, self.q)
        return self.crt(decrypt_to_p, decrypt_to_q)

    def h_function(self, x, xsquare):
        """Computes the h-function as defined in Paillier's paper page 12,
        'Decryption using Chinese-remaindering'.
        """
        return invert(self.l_function(powmod(self.public_key.g, x - 1, xsquare), x), x)

    def l_function(self, x, p):
        """Computes the L function as defined in Paillier's paper. That is: L(x,p) = (x-1)/p"""
        return (x - 1) // p

    def crt(self, mp, mq):
        """The Chinese Remainder Theorem as needed for decryption. Returns the solution modulo n=pq.

        Args:
           mp(int): the solution modulo p.
           mq(int): the solution modulo q.
       """
        u = np.mod((mq - mp) * self.p_inverse, self.q)
        return mp + (u * self.p)

    def __eq__(self, other):
        return self.p == other.p and self.q == other.q

    def __hash__(self):
        return hash((self.p, self.q))


class EncryptedNumber(object):
    """Represents the Paillier encryption of a float or int.

    Typically, an `EncryptedNumber` is created by
    :meth:`PaillierPublicKey.encrypt`. You would only instantiate an
    `EncryptedNumber` manually if you are de-serializing a number
    someone else encrypted.


    Paillier encryption is only defined for non-negative integers less
    than :attr:`PaillierPublicKey.n`. :class:`EncodedNumber` provides
    an encoding scheme for floating point and signed integers that is
    compatible with the partially homomorphic properties of the Paillier
    cryptosystem:

    1. D(E(a) * E(b)) = a + b
    2. D(E(a)**b)     = a * b

    where `a` and `b` are ints or floats, `E` represents encoding then
    encryption, and `D` represents decryption then decoding.

    Args:
      public_key (PaillierPublicKey): the :class:`PaillierPublicKey`
        against which the number was encrypted.
      ciphertext (int): encrypted representation of the encoded number.
      exponent (int): used by :class:`EncodedNumber` to keep track of
        fixed precision. Usually negative.

    Attributes:
      public_key (PaillierPublicKey): the :class:`PaillierPublicKey`
        against which the number was encrypted.
      exponent (int): used by :class:`EncodedNumber` to keep track of
        fixed precision. Usually negative.

    Raises:
      TypeError: if *ciphertext* is not an int, or if *public_key* is
        not a :class:`PaillierPublicKey`.
    """

    # T = EncryptedNumber(self.public_key, self.ciphertext(False).T, self.exponent.T)
    def __init__(self, public_key, ciphertext, exponent=0):
        self.public_key = public_key
        self.__ciphertext = ciphertext
        self.exponent = exponent
        self.__is_obfuscated = False

        if isinstance(self.ciphertext, EncryptedNumber):
            raise TypeError('ciphertext should be an integer')
        if not isinstance(self.public_key, PaillierPublicKey):
            raise TypeError('public_key should be a PaillierPublicKey')

    @property
    def T(self):
        """
        Transposed matrix,this function is eq np.array(data).T
        Examples:
            data = EncryptedNumber([[1,2],[3,4]])
            will return new data = EncryptedNumber([[1,3],[2,4]])
        Returns:

        """
        if isinstance(self.ciphertext(False),np.ndarray):
            return self.__class__(self.public_key, self.ciphertext(False).T, self.exponent.T)
        else:
            return self.__class__(self.public_key, self.ciphertext(False), self.exponent)

    @property
    def shape(self):
        if isinstance(self.exponent, np.ndarray):
            return self.exponent.shape
        else:
            raise TypeError("no shape！maybe (1,1) ,you can use np.ndarray")

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            if other.shape != self.ciphertext(False).shape:
                raise ValueError(f'shape mismatch {other.shape}!={self.ciphertext(False).shape}')

        if isinstance(other, float) or isinstance(other, int):
            if isinstance(self.ciphertext(False), np.ndarray):
                other = np.full(shape=self.ciphertext(False).shape, fill_value=other)

        """Add an int, float, `EncryptedNumber` or `EncodedNumber`."""
        if isinstance(other, EncryptedNumber):
            return self._add_encrypted(other)
        elif isinstance(other, EncodedNumber):
            return self._add_encoded(other)
        else:
            return self._add_scalar(other)

    def __radd__(self, other):
        """Called when Python evaluates `34 + <EncryptedNumber>`
        Required for builtin `sum` to work.
        """
        return self.__add__(other)

    def __mul__(self, other):

        if isinstance(other, np.ndarray):
            if other.shape != self.ciphertext(False).shape:
                raise ValueError(f'shape mismatch {other.shape}!={self.ciphertext(False).shape}')

        if isinstance(other, float) or isinstance(other, int):
            if isinstance(self.ciphertext(False), np.ndarray):
                other = np.full(shape=self.ciphertext(False).shape, fill_value=other)

        """Multiply by an int, float, or EncodedNumber."""
        if isinstance(other, EncryptedNumber):
            raise NotImplementedError('Good luck with that...')
        if isinstance(other, EncodedNumber):
            encoding = other
        else:
            encoding = EncodedNumber.encode(self.public_key, other)
        product = self._raw_mul(encoding.encoding)
        exponent = self.exponent + encoding.exponent

        return EncryptedNumber(self.public_key, product, exponent)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self + (other * -1)

    def __rsub__(self, other):
        return other + (self * -1)

    def __truediv__(self, scalar):
        return self.__mul__(1 / scalar)

    def ciphertext(self, be_secure=True):
        """Return the ciphertext of the EncryptedNumber.

        Choosing a random number is slow. Therefore, methods like
        :meth:`__add__` and :meth:`__mul__` take a shortcut and do not
        follow Paillier encryption fully - every encrypted sum or
        product should be multiplied by r **
        :attr:`~PaillierPublicKey.n` for random r < n (i.e., the result
        is obfuscated). Not obfuscating provides a big speed up in,
        e.g., an encrypted dot product: each of the product terms need
        not be obfuscated, since only the final sum is shared with
        others - only this final sum needs to be obfuscated.

        Not obfuscating is OK for internal use, where you are happy for
        your own computer to know the scalars you've been adding and
        multiplying to the original ciphertext. But this is *not* OK if
        you're going to be sharing the new ciphertext with anyone else.

        So, by default, this method returns an obfuscated ciphertext -
        obfuscating it if necessary. If instead you set `be_secure=False`
        then the ciphertext will be returned, regardless of whether it
        has already been obfuscated. We thought that this approach,
        while a little awkward, yields a safe default while preserving
        the option for high performance.

        Args:
          be_secure (bool): If any untrusted parties will see the
            returned ciphertext, then this should be True.

        Returns:
          an int, the ciphertext. If `be_secure=False` then it might be
            possible for attackers to deduce numbers involved in
            calculating this ciphertext.
        """
        if be_secure and not self.__is_obfuscated:
            self.obfuscate()

        return self.__ciphertext

    def decrease_exponent_to(self, new_exp):
        """Return an EncryptedNumber with same value but lower exponent.

        If we multiply the encoded value by :attr:`EncodedNumber.BASE` and
        decrement :attr:`exponent`, then the decoded value does not change.
        Thus we can almost arbitrarily ratchet down the exponent of an
        `EncryptedNumber` - we only run into trouble when the encoded
        integer overflows. There may not be a warning if this happens.

        When adding `EncryptedNumber` instances, their exponents must
        match.

        This method is also useful for hiding information about the
        precision of numbers - e.g. a protocol can fix the exponent of
        all transmitted `EncryptedNumber` instances to some lower bound(s).

        Args:
          new_exp (int): the desired exponent.

        Returns:
          EncryptedNumber: Instance with the same plaintext and desired
            exponent.

        Raises:
          ValueError: You tried to increase the exponent.
        """
        if np.any(new_exp > self.exponent):
            raise ValueError('New exponent %i should be more negative than '
                             'old exponent %i' % (new_exp, self.exponent))
        if isinstance(self.exponent, np.ndarray):
            other = pow(EncodedNumber.BASE, self.exponent - new_exp)
            # What's the base-2 exponent of the least significant bit?
            # The least significant bit has value 2 ** bin_lsb_exponent
            # bin_lsb_exponent = bin_flt_exponent - cls.FLOAT_MANTISSA_BITS
            #
            # # What's the corresponding base BASE exponent? Round that down.
            # prec_exponent = np.floor(bin_lsb_exponent / cls.LOG2_BASE)
            multiplied = self * other
            multiplied.exponent = new_exp

        else:
            multiplied = self * pow(EncodedNumber.BASE, self.exponent - new_exp)
            multiplied.exponent = new_exp

        return multiplied

    def obfuscate(self):
        """Disguise ciphertext by multiplying by r ** n with random r.

        This operation must be performed for every `EncryptedNumber`
        that is sent to an untrusted party, otherwise eavesdroppers
        might deduce relationships between this and an antecedent
        `EncryptedNumber`.

        For example::

            enc = public_key.encrypt(1337)
            send_to_nsa(enc)       # NSA can't decrypt (we hope!)
            product = enc * 3.14
            send_to_nsa(product)   # NSA can deduce 3.14 by bruteforce attack
            product2 = enc * 2.718
            product2.obfuscate()
            send_to_nsa(product)   # NSA can't deduce 2.718 by bruteforce attack
        """
        r = self.public_key.get_random_lt_n()
        r_pow_n = powmod(r, self.public_key.n, self.public_key.nsquare)
        self.__ciphertext = np.mod(self.__ciphertext * r_pow_n, self.public_key.nsquare)
        self.__is_obfuscated = True

    def _add_scalar(self, scalar):
        """Returns E(a + b), given self=E(a) and b.

        Args:
          scalar: an int or float b, to be added to `self`.

        Returns:
          EncryptedNumber: E(a + b), calculated by encrypting b and
            taking the product of E(a) and E(b) modulo
            :attr:`~PaillierPublicKey.n` ** 2.

        Raises:
          ValueError: if scalar is out of range or precision.
        """

        encoded = EncodedNumber.encode(self.public_key, scalar,
                                       max_exponent=self.exponent)

        return self._add_encoded(encoded)

    def _add_encoded(self, encoded):
        """Returns E(a + b), given self=E(a) and b.

        Args:
          encoded (EncodedNumber): an :class:`EncodedNumber` to be added
            to `self`.

        Returns:
          EncryptedNumber: E(a + b), calculated by encrypting b and
            taking the product of E(a) and E(b) modulo
            :attr:`~PaillierPublicKey.n` ** 2.

        Raises:
          ValueError: if scalar is out of range or precision.
        """
        if self.public_key != encoded.public_key:
            raise ValueError("Attempted to add numbers encoded against "
                             "different public keys!")

        # In order to add two numbers, their exponents must match.

        a, b = self, encoded
        if isinstance(a.exponent, np.ndarray):
            mins = np.minimum(a.exponent, b.exponent)
            a = self.decrease_exponent_to(mins)
            b = b.decrease_exponent_to(mins)
        else:
            if a.exponent > b.exponent:
                a = self.decrease_exponent_to(b.exponent)
            elif a.exponent < b.exponent:
                b = b.decrease_exponent_to(a.exponent)

        # Don't bother to salt/obfuscate in a basic operation, do it
        # just before leaving the computer.
        encrypted_scalar = a.public_key.raw_encrypt(b.encoding, 1)
        sum_ciphertext = a._raw_add(a.ciphertext(False), encrypted_scalar)
        return EncryptedNumber(a.public_key, sum_ciphertext, a.exponent)

    def _add_encrypted(self, other):
        """Returns E(a + b) given E(a) and E(b).

        Args:
          other (EncryptedNumber): an `EncryptedNumber` to add to self.

        Returns:
          EncryptedNumber: E(a + b), calculated by taking the product
            of E(a) and E(b) modulo :attr:`~PaillierPublicKey.n` ** 2.

        Raises:
          ValueError: if numbers were encrypted against different keys.
        """
        if self.public_key != other.public_key:
            raise ValueError("Attempted to add numbers encrypted against "
                             "different public keys!")

        # In order to add two numbers, their exponents must match.
        a, b = self, other
        if isinstance(a.exponent, np.ndarray) and isinstance(b.exponent, np.ndarray):
            if a.exponent.shape!=b.exponent.shape:
                raise ValueError("x+y:x.shape is not eq y.shape")
            mins = np.minimum(a.exponent, b.exponent)
            a = self.decrease_exponent_to(mins)
            b = b.decrease_exponent_to(mins)
        elif not isinstance(a.exponent, np.ndarray) and not isinstance(b.exponent, np.ndarray):
            if a.exponent > b.exponent:
                a = self.decrease_exponent_to(b.exponent)
            elif a.exponent < b.exponent:
                b = b.decrease_exponent_to(a.exponent)
        else:
            if not isinstance(a.exponent, np.ndarray):
                a = EncryptedNumber(a.public_key,ciphertext=np.full(shape=b.shape,fill_value=a.ciphertext(False))
                                    ,exponent=np.full(shape=b.shape,fill_value=a.exponent))
            else:
                b = EncryptedNumber(b.public_key,ciphertext=np.full(shape=a.shape,fill_value=b.ciphertext(False))
                                    ,exponent=np.full(shape=a.shape,fill_value=b.exponent))


                mins = np.minimum(a.exponent, b.exponent)
                a = self.decrease_exponent_to(mins)
                b = b.decrease_exponent_to(mins)
        sum_ciphertext = a._raw_add(a.ciphertext(False), b.ciphertext(False))
        return EncryptedNumber(a.public_key, sum_ciphertext, a.exponent)

    def _raw_add(self, e_a, e_b):
        """Returns the integer E(a + b) given ints E(a) and E(b).

        N.B. this returns an int, not an `EncryptedNumber`, and ignores
        :attr:`ciphertext`

        Args:
          e_a (int): E(a), first term
          e_b (int): E(b), second term

        Returns:
          int: E(a + b), calculated by taking the product of E(a) and
            E(b) modulo :attr:`~PaillierPublicKey.n` ** 2.
        """
        return e_a * e_b % self.public_key.nsquare

    def _raw_mul(self, plaintext):
        """Returns the integer E(a * plaintext), where E(a) = ciphertext

        Args:
          plaintext (int): number by which to multiply the
            `EncryptedNumber`. *plaintext* is typically an encoding.
            0 <= *plaintext* < :attr:`~PaillierPublicKey.n`

        Returns:
          int: Encryption of the product of `self` and the scalar
            encoded in *plaintext*.

        Raises:
          TypeError: if *plaintext* is not an int.
          ValueError: if *plaintext* is not between 0 and
            :attr:`PaillierPublicKey.n`.
        """
        if not (isinstance(plaintext, int) or isinstance(plaintext, np.ndarray)):
            raise TypeError('Expected ciphertext to be int, not %s' %
                            type(plaintext))

        if np.any(plaintext < 0) or np.any(plaintext >= self.public_key.n):
            raise ValueError('Scalar out of bounds: %i' % plaintext)
        if np.any(self.public_key.n - self.public_key.max_int <= plaintext):
            # Very large plaintext, play a sneaky trick using inverses
            neg_c = np.frompyfunc(invert, 2, 1)(self.ciphertext(False), self.public_key.nsquare)
            neg_scalar = self.public_key.n - plaintext
            return np.frompyfunc(powmod, 3, 1)(neg_c, neg_scalar, self.public_key.nsquare)
        else:
            return np.frompyfunc(powmod, 3, 1)(self.ciphertext(False), plaintext, self.public_key.nsquare)

    def toArray(self):
        """
        EncryptedNumber matrix (object) conversion to EncryptedNumber matrix (np.ndarray)
        :param :
        :return: np.ndarray
        """
        data = np.frompyfunc(lambda x, y, z: EncryptedNumber(x, y, z), 3, 1)(
            self.public_key, self.ciphertext(False), self.exponent)
        return data


    def sum(self):
        """
        EncryptedNumber matrix (object) get sum
          return EncryptedNumber object

        example:
              Z[A] = EncryptedNumber()
              A = np.array([[1,2,3],
                           [1,2,3],
                           [1,2,3]])

              Z[A].sum() = [A.sum()]=[18]

        :param:
        :return: EncryptedNumber()
        """

        new_exp = self.exponent.min()
        ml = self.__mul__(pow(EncodedNumber.BASE, self.exponent - new_exp))
        ml.exponent = np.full(fill_value=new_exp, shape=self.exponent.shape)
        sum_ciphertext = reduce(lambda x, y: x + y, ml.toArray().flatten())
        return sum_ciphertext

    def dot(self, other, is_pool=False):
        """
        The ciphertext matrix is multiplied by the plaintext matrix

              input: matrix_B:np.ndarray
              return EncryptedNumber()
              example:
                  Z[A] = EncryptedNumber()
                  A = np.array([[1,2,3],
                               [1,2,1]])

                  B = np.array([[1,1],
                               [1,2],
                               [0,3]])
                  Z[C]=Z[A].dot(B)=[[3,14],
                                  [3,8]]
        :param: other:np.ndarray
        :return: EncryptedNumber()
        """
        if isinstance(other,EncryptedNumber):
            if self.public_key!=other.public_key:
                raise ValueError("x dot y: public_key x is not eq to public_key y ! So Do you believe in light?")
            else:
                raise ValueError("x dot y: Even though public_key x is eq to public_key y, "
                                 "I don't want to write this code, "
                                 "so you'd better multiply the plaintext of A and B and encrypt it.")
        if is_pool:
            return self.dot_pool(other)
        else:
            new_base = []
            new_exponent = []
            for i in range(self.exponent.shape[0]):
                for j in range(other.shape[1]):
                    new_data = EncryptedNumber(self.public_key,
                                               self.ciphertext(False)[i],
                                               self.exponent[i]).__mul__(other[:, j]).sum()

                    new_base.append(new_data.ciphertext(False))
                    new_exponent.append(new_data.exponent)
            new_base = np.array(new_base).reshape(self.exponent.shape[0], other.shape[1])
            new_exponent = np.array(new_exponent).reshape(self.exponent.shape[0], other.shape[1])
            return EncryptedNumber(self.public_key, new_base, new_exponent)

    def mulsum(self, lists):
        """
        get sum
        Args:
            lists:

        Returns:

        """
        new_data = lists[0].__mul__(lists[1]).sum()
        return [new_data.exponent, new_data.ciphertext(False)]

    def dot_pool(self, other):
        """
        do dot by pool
        Args:
            other: np.ndarray

        Returns:EncryptedNumber

        """
        lists = []
        for i in range(self.exponent.shape[0]):
            for j in range(other.shape[1]):
                lists.append([EncryptedNumber(self.public_key,
                                              self.ciphertext(False)[i],
                                              self.exponent[i]), other[:, j]])
        pool = mp.Pool(processes=mp.cpu_count()-1)
        result = np.array(pool.map(self.mulsum, lists)).reshape(self.exponent.shape[0], other.shape[1], 2)
        pool.close()
        pool.join()
        results = EncryptedNumber(self.public_key, result[:, :, 1], result[:, :, 0])
        return results
