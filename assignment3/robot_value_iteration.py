# Initialization
alpha = 0.25
beta = 0.25
r_search = 5
r_wait = 1
gamma = 0.8
threshold = 0.00001

# Initial values
V = {
    "high": 0,
    "low": 0
}
delta = float('inf')

# Value iteration
iteration = 0
while delta > threshold:
    iteration += 1
    V_prev = V.copy()

    # Update for high state
    q_search_high = alpha * (r_search + gamma * V_prev["high"]) + (1 - alpha) * (r_search + gamma * V_prev["low"])
    q_wait_high = r_wait + gamma * V_prev["high"]

    # Update for low state
    q_search_low = (1 - beta) * (-3 + gamma * V_prev["high"]) + beta * (r_search + gamma * V_prev["low"])
    q_wait_low = r_wait + gamma * V_prev["low"]
    q_recharge_low = gamma * V_prev["high"]

    # Update state values
    V["high"] = max(q_search_high, q_wait_high)
    V["low"] = max(q_search_low, q_wait_low, q_recharge_low)

    delta = max(abs(V["high"] - V_prev["high"]), abs(V["low"] - V_prev["low"]))
    if(iteration<=2):
        print(f"Iteration {iteration}:")
        print(f"State values: {V}")
        print("-----------------------")

# Extracting the optimal policy
pi = {
    "high": "search" if q_search_high > q_wait_high else "wait",
    "low": max([("search", q_search_low), ("wait", q_wait_low), ("recharge", q_recharge_low)], key=lambda x: x[1])[0]
}


print("final State Values:", V)
