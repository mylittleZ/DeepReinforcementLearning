# Initialization
alpha = 0.25
beta = 0.25
r_search = 5
r_wait = 1
gamma = 0.8
threshold = 0.00001

# Stochastic policies
pi = {
    "high": {"search": 0.5, "wait": 0.5},
    "low": {"search": 0.5, "wait": 0.25, "recharge": 0.25}
}

# Initial values
V = {
    "high": 0,
    "low": 0
}

def evaluate_policy(pi, V):
    while True:
        delta = 0
        old_V = V.copy()

        # Update value for 'high' state
        V["high"] = pi["high"]["search"] * (alpha * (r_search + gamma * old_V["high"]) + (1 - alpha) * (r_search + gamma * old_V["low"])) + pi["high"]["wait"] * (r_wait + gamma * old_V["high"])

        # Update value for 'low' state
        V["low"] = pi["low"]["search"] * ((1 - beta) * (-3 + gamma * old_V["high"]) + beta * (r_search + gamma * old_V["low"])) + pi["low"]["wait"] * (r_wait + gamma * old_V["low"]) + pi["low"]["recharge"] * gamma * old_V["high"]

        delta = max(delta, abs(V["high"] - old_V["high"]), abs(V["low"] - old_V["low"]))
        if delta < threshold:
            break

def improve_policy(pi, V):
    actions_high = ["search", "wait"]
    actions_low = ["search", "wait", "recharge"]

    # Compute q-values for 'high' state
    q_high = {
        "search": alpha * (r_search + gamma * V["high"]) + (1 - alpha) * (r_search + gamma * V["low"]),
        "wait": r_wait + gamma * V["high"]
    }

    # Compute q-values for 'low' state
    q_low = {
        "search": (1 - beta) * (-3 + gamma * V["high"]) + beta * (r_search + gamma * V["low"]),
        "wait": r_wait + gamma * V["low"],
        "recharge": gamma * V["high"]
    }

    # Update policies based on maximum q-value
    best_action_high = max(actions_high, key=q_high.get)
    best_action_low = max(actions_low, key=q_low.get)

    pi["high"] = {action: int(action == best_action_high) for action in actions_high}
    pi["low"] = {action: int(action == best_action_low) for action in actions_low}

    return pi


# Main loop for policy iteration
is_policy_stable = False
iteration = 0
while not is_policy_stable:
    old_V = V.copy()
    evaluate_policy(pi, V)
    old_pi = {state: {action: prob for action, prob in pi[state].items()} for state in pi}
    iteration += 1
    print(f"Iteration {iteration}:")
    print(f"Policy: {pi}")
    print(f"State values: {V}")
    print("-----------------------")
    improve_policy(pi, V)

    if all(old_pi[state] == pi[state] for state in pi) or max([abs(V[s] - old_V[s]) for s in V]) < threshold:
        is_policy_stable = True


print("Optimal Policy:", pi)
print("Optimal State Values:", V)
