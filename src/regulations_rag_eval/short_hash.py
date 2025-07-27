import hashlib

def generate_short_hash(input_string: str) -> str:
    """
    Generates a 15-character hash from a string using MD5 and
    truncating the hexadecimal output.


    Args:
        input_string: The string to hash.

    Returns:
        A 15-character hexadecimal string derived from the MD5 hash.

    """
    if not isinstance(input_string, str):
        raise TypeError("Input must be a string.")

    # Encode the string to bytes (UTF-8 is a standard choice)
    string_bytes = input_string.encode('utf-8')

    # Create an MD5 hash object
    md5_hasher = hashlib.md5()

    # Update the hasher with the input bytes
    md5_hasher.update(string_bytes)

    # Get the full hexadecimal digest (32 characters for MD5)
    full_hex_hash = md5_hasher.hexdigest()

    # Truncate to the first 15 characters
    short_hash = full_hex_hash[:15]

    return short_hash
