import numpy as np

def random_sample_greedy_search(q_table, states, sample_size, epsilon, seed):
    rng = np.random.RandomState(seed)

    num_actions = q_table.shape[1]
    k = min(sample_size, num_actions)

    actions = []
    q_values = []
    matches = 0

    for state in states:
        if rng.random() < epsilon:
            action = rng.randint(0, num_actions)

        else:
            sampled = rng.choice(num_actions, size=k, replace=False)
            sampled.sort()

            action = sampled[np.argmax(q_table[state, sampled])]

        actions.append(int(action))
        q_values.append(round(float(q_table[state, action]), 4))

        true_greedy = np.argmax(q_table[state])

        if action == true_greedy:
            matches += 1

    match_rate = matches / len(states) if len(states) > 0 else 0.0

    return {
        'actions': actions,
        'q_values': q_values,
        'greedy_match_rate': round(match_rate, 4)
    }