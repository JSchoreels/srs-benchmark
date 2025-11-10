import json
import warnings

import numpy as np
import scipy

warnings.filterwarnings('ignore')


def logp_wilcox(x, y, correction=False):
    # method='wilcox'
    # mode='approx'
    # alternative='two-sided'
    assert len(x) == len(y)
    x = np.asarray(x)
    y = np.asarray(y)

    def rankdata(a, method="average"):
        a = np.asarray(a)
        if a.size == 0:
            return np.empty(a.shape)
        sorter = np.argsort(a)
        inv = np.empty(sorter.size, dtype=np.intp)
        inv[sorter] = np.arange(sorter.size, dtype=np.intp)

        if method == "ordinal":
            result = inv + 1
        else:
            a = a[sorter]
            obs = np.r_[True, a[1:] != a[:-1]]
            dense = obs.cumsum()[inv]

            if method == "dense":
                result = dense
            else:
                # cumulative counts of each unique value
                count = np.r_[np.nonzero(obs)[0], len(obs)]
                if method == "max":
                    result = count[dense]
                elif method == "min":
                    result = count[dense - 1] + 1
                elif method == "average":
                    result = 0.5 * (count[dense] + count[dense - 1] + 1)

        return result

    diff = x - y
    count = diff.size

    ranks = rankdata(abs(diff))
    r_plus = np.sum((diff > 0) * ranks)
    r_minus = np.sum((diff < 0) * ranks)
    if r_plus > r_minus:
        # x is greater than y
        which_one = 0
    else:
        # y is greater than x
        which_one = 1

    T = min(r_plus, r_minus)

    mn = count * (count + 1.0) * 0.25
    se = count * (count + 1.0) * (2.0 * count + 1.0)

    replist, repnum = scipy.stats.find_repeats(ranks)
    if repnum.size != 0:
        # correction for repeated elements.
        se -= 0.5 * (repnum * (repnum * repnum - 1)).sum()

    se = np.sqrt(se / 24)

    # apply continuity correction if applicable
    d = 0
    if correction:
        d = 0.5 * np.sign(T - mn)

    # compute statistic
    z = (T - mn - d) / se
    if abs(z) > 37:
        a = 0.62562732
        b = 0.22875463
        logp_approx = np.log1p(-np.exp(-a * abs(z))) - np.log(abs(z)) - (z**2) / 2 - b
    else:
        logp_approx = np.log(2.0 * scipy.stats.norm.sf(abs(z)))

    # returns the decimal logarithm of the p-value
    return np.log10(np.e) * logp_approx, which_one


results_path = 'result'
baseline_name = 'FSRS-6-recency-old.jsonl'  # FSRS-6-recency-old for excluding same-day reviews
comparison_name = 'FSRS-6-recency-inflection.jsonl'
baseline_file = results_path + '/' + baseline_name
comparison_file = results_path + '/' + comparison_name


# metric = 'RMSE(bins)'  # 'RMSE(bins)', 'LogLoss'
unweighted = True
# for metric in ['RMSE(bins)', 'LogLoss']:
for metric in ['LogLoss']:
    max_user = 1000

    dictionary_metric = {}
    dictionary_sizes = {}
    with open(baseline_file, "r") as f:
        data = f.readlines()
        common_set = set([json.loads(x)["user"] for x in f.readlines() if int(json.loads(x)["user"]) <= max_user])

    last_x = 0
    for x in common_set:
        if x > last_x + 1:
            print(f'User {last_x + 1} is missing')
        # else:
        #     print(x, last_x)
        last_x = x
    # print('')

    data = [json.loads(x) for x in data]

    for result in data:
        value = result["metrics"][metric]
        user = result["user"]
        size = result["size"]
        if len(dictionary_metric) < max_user:
            dictionary_metric.update({user: value})
            if unweighted:
                dictionary_sizes.update({user: 1})
            else:
                dictionary_sizes.update({user: size})

    sorted_dictionary_LogLoss = dict(sorted(dictionary_metric.items()))
    LogLoss_baseline = list(sorted_dictionary_LogLoss.values())
    sorted_dictionary_sizes = dict(sorted(dictionary_sizes.items()))
    sizes = list(sorted_dictionary_sizes.values())

    dictionary_metric2 = {}
    dictionary_sizes2 = {}
    with open(comparison_file, "r") as f:
        data2 = f.readlines()
    data2 = [json.loads(x) for x in data2]
    for result2 in data2:
        value = result2["metrics"][metric]
        user = result2["user"]
        size = result2["size"]
        if len(dictionary_metric2) < max_user:
            dictionary_metric2.update({user: value})
            if unweighted:
                dictionary_sizes2.update({user: 1})
            else:
                dictionary_sizes2.update({user: size})

    sorted_dictionary_LogLoss2 = dict(sorted(dictionary_metric2.items()))
    LogLoss_comparison = list(sorted_dictionary_LogLoss2.values())
    sorted_dictionary_sizes2 = dict(sorted(dictionary_sizes2.items()))
    sizes2 = list(sorted_dictionary_sizes2.values())

    # Sanity check
    assert len(sizes) == len(sizes2), f'{len(sizes)}, {len(sizes2)}'

    # all_users_1 = list(range(1, max_user+1))
    # all_users_2 = list(range(1, max_user+1))
    # for result1, result2 in zip(data, data2):
    #     user1 = result1["user"]
    #     user2 = result2["user"]
    #     all_users_1.remove(user1)
    #     all_users_2.remove(user2)
    #
    # if len(all_users_1) > 0:
    #     print(f'Missing users (baseline): {all_users_1}')
    #
    # if len(all_users_2) > 0:
    #     print(f'Missing users (comparison): {all_users_2}')

    assert sum(sizes) == sum(sizes2), f'{sum(sizes)}, {sum(sizes2)}'

    print(f'Mean of the baseline={np.average(LogLoss_baseline, weights=sizes):.4f}')
    print(f'Mean of the comparison={np.average(LogLoss_comparison, weights=sizes):.4f}')
    # print(f'99th percentile LogLoss (baseline)={np.percentile(LogLoss_baseline, 99):.4f}')
    # print(f'99th percentile LogLoss (comparison)={np.percentile(LogLoss_comparison, 99):.4f}')

    counter = 0
    for n in range(len(LogLoss_baseline)):
        if LogLoss_baseline[n] < LogLoss_comparison[n]:
            counter += 1

    print(f'Percentage of users who would be better off using the baseline: {100*counter/len(LogLoss_baseline):.1f}%')

    print('')
    logp, which_one = logp_wilcox(LogLoss_baseline, LogLoss_comparison)
    if which_one == 0:
        print('Baseline is worse')
    else:
        print('Comparison is worse')
    wilcox = scipy.stats.wilcoxon(LogLoss_baseline, LogLoss_comparison).pvalue  # dependent (paired)

    if wilcox <= 0.001:
        print(f'Wilcoxon signed-rank test={wilcox}')
    else:
        print(f'Wilcoxon signed-rank test={wilcox:.4f}')

    if wilcox == -1:
        print(f'This test is not applicable in this case')
    if 0 <= wilcox < 0.001:
        print('p<0.001')
    elif 0 <= wilcox < 0.01:
        print('p<0.01')
    elif wilcox >= 0.01:
        print('Not statistically significant')

    print('----------------------------')