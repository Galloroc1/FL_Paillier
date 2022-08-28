import random
import os

from encoding import EncodedNumber
from util import invert, powmod, mpz
import numpy as np

# from functools import wraps
SUPPORT_TYPE = ["float", "int", "numpy.ndarray"]
DEFAULT_KEYSIZE = 1024


class PublicKey(object):

    def __init__(self, n):
        self.g = n + 1
        self.n = n
        self.nsquare = n * n
        self.max_int = n // 3 - 1

    def __repr__(self):
        publicKeyHash = hex(hash(self))[2:]
        return "<PublicKey {}>".format(publicKeyHash[:10])

    def __eq__(self, other):
        return self.n == other.n

    def __hash__(self):
        return hash(self.n)

    def __str__(self):

        info = ""
        info = info + "id : " + str(id(self)) + "\n"
        info = info + "processing : " + str(os.getpid()) + "\n"
        info = info + "n : " + str(self.n)[0:10] + "..." + "\n"
        return info

    def apply_obfuscator(self, ciphertext, random_value=None):
        """
        """
        r = random_value or random.SystemRandom().randrange(1, self.n)
        obfuscator = powmod(r, self.n, self.nsquare)

        return (ciphertext * obfuscator) % self.nsquare

    def get_ciphertext(self, plaintext):
        if plaintext >= (self.n - self.max_int) and plaintext < self.n:
            # Very large plaintext, take a sneaky shortcut using inverses
            neg_plaintext = self.n - plaintext  # = abs(plaintext - nsquare)
            neg_ciphertext = (self.n * neg_plaintext + 1) % self.nsquare
            ciphertext = invert(neg_ciphertext, self.nsquare)
        else:
            ciphertext = (self.n * plaintext + 1) % self.nsquare

        return ciphertext

    def raw_encrypt(self, plaintext, random_value=None):
        assert isinstance(plaintext, np.ndarray) or isinstance(plaintext, int), \
            f"not support value type {type(plaintext)}"
        if isinstance(plaintext, np.ndarray):
            ciphertext = np.frompyfunc(self.get_ciphertext, 1, 1)(plaintext)
        else:
            ciphertext = self.get_ciphertext(plaintext)
        ciphertext = self.apply_obfuscator(ciphertext, random_value)

        return ciphertext

    def encrypt(self, value, precision=None, random_value=None,rtype='None'):
        """Encode and Paillier encrypt a real number value.
            rtype:return_value's type,{'ndarray','None'},
                if None then return EncryptedNumber ,if ndarray then return ndarray
        """
        if isinstance(value, EncodedNumber):
            value = value.decode()
        encoding = EncodedNumber.encode(value, self.n, precision, self.max_int)
        obfuscator = random_value or 1
        ciphertext = self.raw_encrypt(encoding.encoding, random_value=obfuscator)
        encryptednumber = EncryptedNumber(self, ciphertext, encoding.exponent)
        #
        if random_value is None:
            encryptednumber.apply_obfuscator()
        if rtype == 'ndarray':
            encryptednumber = toArray(encryptednumber)
        return encryptednumber


class EncryptedNumber(object):
    """Represents the Paillier encryption of a float or int.
    """
    __slots__ = ['public_key', '__ciphertext', 'exponent', '__is_obfuscator']

    def __init__(self, public_key, ciphertext, exponent=0):
        self.public_key = public_key
        self.__ciphertext = ciphertext
        self.exponent = exponent
        self.__is_obfuscator = False

        if not isinstance(self.__ciphertext, int) and not isinstance(self.__ciphertext, np.ndarray):
            raise TypeError("ciphertext should be an int, not: %s" % type(self.__ciphertext))

        if not isinstance(self.public_key, PublicKey):
            raise TypeError("public_key should be a PublicKey, not: %s" % type(self.public_key))

    def ciphertext(self, be_secure=True):
        """return the ciphertext of the PaillierEncryptedNumber.
        """
        if be_secure and not self.__is_obfuscator:
            self.apply_obfuscator()

        return self.__ciphertext

    def apply_obfuscator(self):
        """ciphertext by multiplying by r ** n with random r
        """
        self.__ciphertext = self.public_key.apply_obfuscator(self.__ciphertext)
        self.__is_obfuscator = True

    def __add__(self, other):
        if isinstance(other, EncryptedNumber):
            return self.__add_encryptednumber(other)
        else:
            return self.__add_scalar(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + (other * -1)

    def __rsub__(self, other):
        return other + (self * -1)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        return self.__mul__(1 / scalar)

    def __mul__(self, scalar):
        """return Multiply by an scalar(such as int, float)
        """
        if isinstance(scalar, EncodedNumber):
            scalar = scalar.decode()
        encode = EncodedNumber.encode(scalar, self.public_key.n, self.public_key.max_int)
        plaintext = encode.encoding

        if plaintext < 0 or plaintext >= self.public_key.n:
            raise ValueError("Scalar out of bounds: %i" % plaintext)

        if plaintext >= self.public_key.n - self.public_key.max_int:
            # Very large plaintext, play a sneaky trick using inverses
            neg_c = invert(self.ciphertext(False), self.public_key.nsquare)
            neg_scalar = self.public_key.n - plaintext
            ciphertext = powmod(neg_c, neg_scalar, self.public_key.nsquare)
        else:
            ciphertext = powmod(self.ciphertext(False), plaintext, self.public_key.nsquare)

        exponent = self.exponent + encode.exponent

        return EncryptedNumber(self.public_key, ciphertext, exponent)

    def increase_exponent_to(self, new_exponent):
        """return PaillierEncryptedNumber:
           new PaillierEncryptedNumber with same value but having great exponent.
        """
        if new_exponent < self.exponent:
            raise ValueError("New exponent %i should be great than old exponent %i" % (new_exponent, self.exponent))

        factor = pow(EncodedNumber.BASE, new_exponent - self.exponent)
        new_encryptednumber = self.__mul__(factor)
        new_encryptednumber.exponent = new_exponent

        return new_encryptednumber

    def __align_exponent(self, x, y):
        """return x,y with same exponet
        """
        if x.exponent < y.exponent:
            x = x.increase_exponent_to(y.exponent)
        elif x.exponent > y.exponent:
            y = y.increase_exponent_to(x.exponent)

        return x, y

    def __add_scalar(self, scalar):
        """return PaillierEncryptedNumber: z = E(x) + y
        """
        if isinstance(scalar, EncodedNumber):
            scalar = scalar.decode()
        encoded = EncodedNumber.encode(scalar,
                                       self.public_key.n,
                                       self.public_key.max_int,
                                       max_int=self.exponent)
        return self.__add_fixpointnumber(encoded)

    def __add_fixpointnumber(self, encoded):
        """return PaillierEncryptedNumber: z = E(x) + FixedPointNumber(y)
        """
        if self.public_key.n != encoded.n:
            raise ValueError("Attempted to add numbers encoded against different public keys!")

        # their exponents must match, and align.
        x, y = self.__align_exponent(self, encoded)

        encrypted_scalar = x.public_key.raw_encrypt(y.encoding, 1)
        encryptednumber = self.__raw_add(x.ciphertext(False), encrypted_scalar, x.exponent)

        return encryptednumber

    def __add_encryptednumber(self, other):
        """return PaillierEncryptedNumber: z = E(x) + E(y)
        """
        if self.public_key != other.public_key:
            raise ValueError("add two numbers have different public key!")

        # their exponents must match, and align.
        x, y = self.__align_exponent(self, other)

        encryptednumber = self.__raw_add(x.ciphertext(False), y.ciphertext(False), x.exponent)

        return encryptednumber

    def __raw_add(self, e_x, e_y, exponent):
        """return the integer E(x + y) given ints E(x) and E(y).
        """
        ciphertext = mpz(e_x) * mpz(e_y) % self.public_key.nsquare

        return EncryptedNumber(self.public_key, int(ciphertext), exponent)


def toArray(x:EncryptedNumber)-> np.ndarray:
    if isinstance(x,np.ndarray):
        print("warning : func : toArray() : param x is array, maybe you should check your data type")
        return x
    return np.frompyfunc(lambda x,y,z:EncryptedNumber(x,y,z),3,1)(x.public_key,x.ciphertext(False),x.exponent)