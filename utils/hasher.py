import hashlib
from utils.enums import HashAlgorithm


class Hasher:
    """ This class is used for computing hash of the file
        using SHA-1 or SHA-256 algorithms
        and comparing hashes of two files. """

    @staticmethod
    def check_hash_equals(algorithm_type, file1, file2):
        """This function compares hashes of two files
                 passed into it using the specified hash algorithm."""
        algorithm = hashlib.sha1 if algorithm_type == HashAlgorithm.SHA1 else hashlib.sha256
        hash1 = Hasher.__compute_hash(algorithm, file1)
        hash2 = Hasher.__compute_hash(algorithm, file2)
        assert hash1 == hash2

    @staticmethod
    def __compute_hash(algorithm, file):
        """This function returns the hash of the file
            passed into it using the specified hash algorithm."""
        h = algorithm()
        with open(file, 'rb') as f:
            for byte_block in iter(lambda: f.read(1024), b""):
                h.update(byte_block)
        print(h.hexdigest())
        return h.hexdigest()

    @staticmethod
    def compute_hash_sha1(file):
        """This function returns the SHA-1 hash
            of the file passed into it."""
        return Hasher.__compute_hash(hashlib.sha1, file)

    @staticmethod
    def compute_hash_sha256(file):
        """This function returns the SHA-256 hash
            of the file passed into it."""
        return Hasher.__compute_hash(hashlib.sha256, file)
