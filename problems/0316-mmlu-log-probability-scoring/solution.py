import numpy as np

def mmlu_log_prob_score(log_probs: list, correct_answers: list) -> dict:
    pred_answers = []
    for probs in log_probs:
        pred_answers.append(np.argmax(probs))
    correct = sum(
        pred == answer
        for pred, answer in zip(pred_answers, correct_answers)
    )
    accuracy = correct / len(correct_answers)

    # Convert log-probabilities to probabilities
    # Using softmax with numerical stability
    correct_probs = []

    for probs, answer in zip(log_probs, correct_answers):
        probs = np.array(probs)

        # Subtract maximum log-probability for numerical stability
        exp_probs = np.exp(probs - np.max(probs))
        probabilities = exp_probs / np.sum(exp_probs)

        correct_probs.append(probabilities[answer])

    # Average probability assigned to the correct answer
    avg_correct_prob = np.mean(correct_probs)

    return {
        "accuracy": accuracy,
        "predictions": pred_answers,
        "avg_correct_prob": float(avg_correct_prob)
    }