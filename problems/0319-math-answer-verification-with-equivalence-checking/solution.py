import re
import math

def verify_math_answer(predicted: str, ground_truth: str, tol: float = 1e-6) -> bool:
    if predicted == ground_truth:
        return True

    def to_num(s):
        try:
            s = s.replace("^", "**")
            return float(eval(s, {"__builtins__": {}}, {"sqrt": math.sqrt, "pi": math.pi}))
        except Exception:
            return None

    p, g = to_num(predicted), to_num(ground_truth)
    if p is None or g is None:
        return False
    return abs(p - g) <= tol