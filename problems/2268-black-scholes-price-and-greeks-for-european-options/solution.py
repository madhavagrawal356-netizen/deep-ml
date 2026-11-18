import math


def black_scholes(S, K, T, r, sigma, option="call"):
    """Return the Black-Scholes price and Greeks of a European call or put."""
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")

    # Expiry case
    if T == 0:
        if option == "call":
            price = max(S - K, 0)
            delta = 1.0 if S > K else 0.0
        else:
            price = max(K - S, 0)
            delta = -1.0 if S < K else 0.0

        return {
            "price": price,
            "delta": delta,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0
        }

    # Standard normal CDF
    def N(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    # Standard normal PDF
    def phi(x):
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    sqrt_T = math.sqrt(T)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    gamma = phi(d1) / (S * sigma * sqrt_T)
    vega = S * phi(d1) * sqrt_T

    if option == "call":
        price = S * N(d1) - K * math.exp(-r * T) * N(d2)
        delta = N(d1)
        theta = (
            -S * phi(d1) * sigma / (2 * sqrt_T)
            - r * K * math.exp(-r * T) * N(d2)
        )
    else:
        price = K * math.exp(-r * T) * N(-d2) - S * N(-d1)
        delta = N(d1) - 1
        theta = (
            -S * phi(d1) * sigma / (2 * sqrt_T)
            + r * K * math.exp(-r * T) * N(-d2)
        )

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta
    } 
