# Mock SSL module for Pyodide
class SSLContext:
    def __init__(self, *args, **kwargs):
        pass
    def load_cert_chain(self, *args, **kwargs):
        pass
    def wrap_socket(self, sock, *args, **kwargs):
        return sock

def create_default_context(*args, **kwargs):
    return SSLContext()

CERT_NONE = 0
CERT_OPTIONAL = 1
CERT_REQUIRED = 2
PROTOCOL_TLS = 0
PROTOCOL_TLS_CLIENT = 1
PROTOCOL_TLS_SERVER = 2
OP_NO_SSLv2 = 0
OP_NO_SSLv3 = 0
OP_NO_TLSv1 = 0

# Urllib3/Requests compatibility
OPENSSL_VERSION = "OpenSSL 1.1.1  11 Sep 2018"
OPENSSL_VERSION_NUMBER = 269488319
OPENSSL_VERSION_INFO = (1, 1, 1, 0, 15)
HAS_SNI = True
HAS_ECDHE = True

class SSLError(Exception):
    pass
