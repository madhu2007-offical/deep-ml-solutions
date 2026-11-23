import numpy as np

def expected_losses(posterior, loss):
    return posterior @ loss


def bayes_policy(posterior, loss):
    return np.argmin(expected_losses(posterior, loss), axis=1)


def covid_table(prevalence, sens, spec, drug_cost, qaly_young=60.0, qaly_old=10.0):
    p_negative = (1 - sens) * prevalence + spec * (1 - prevalence)
    p_positive = sens * prevalence + (1 - spec) * (1 - prevalence)

    posterior_negative = ((1 - sens) * prevalence) / p_negative
    posterior_positive = (sens * prevalence) / p_positive

    rows = []

    for test, p_covid in [(0, posterior_negative), (1, posterior_positive)]:
        for age, qaly in [(0, qaly_young), (1, qaly_old)]:
            cost_nothing = p_covid * qaly
            cost_drugs = drug_cost

            action = int(cost_drugs < cost_nothing)

            rows.append([
                test,
                age,
                float(p_covid),
                float(cost_nothing),
                float(cost_drugs),
                action
            ])

    return rows