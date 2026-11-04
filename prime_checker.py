def is_prime(n):
    """Checks if a number is prime."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

if __name__ == "__main__":
    test_cases = [
        (2, True),
        (4, False),
        (17, True),
        (25, False),
        (97, True)
    ]

    print(f"{'Number':<10} | {'Expected':<10} | {'Actual':<10} | {'Result'}")
    print("-" * 50)
    for num, expected in test_cases:
        actual = is_prime(num)
        result = "PASS" if actual == expected else "FAIL"
        print(f"{num:<10} | {expected:<10} | {actual:<10} | {result}")
