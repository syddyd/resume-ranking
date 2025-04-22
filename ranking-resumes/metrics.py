'''Evaluation Metrics'''

#Evauate subgroup accuracy
from aif360.metrics import ClassificationMetric


metric = ClassificationMetric(
    aif_data,
    preds,
    privileged_groups=[{'ethnicity': 1}, {'gender': 1}],
    unprivileged_groups=[{'ethnicity': 0}, {'gender': 0}]
)
#   Subgroup metrics
#heatmap
clf.heatmapflag = True
clf.heatmap_path = 'heatmap'
clf.generate_heatmap(aif_data, dataset.labels)
Image(filename='{}.png'.format(clf.heatmap_path))

#blackbox auditing
gerry_metric = BinaryLabelDatasetMetric(aif_data)
gamma_disparity = gerry_metric.rich_subgroup(array_to_tuple(dataset.labels),'FP')
print(gamma_disparity)
#FPR VS FNR data analysis
def fp_vs_fn(dataset, gamma_list, iters):
    fp_auditor = Auditor(dataset, 'FP')
    fn_auditor = Auditor(dataset, 'FN')
    fp_violations = []
    fn_violations = []
    for g in gamma_list:
        print('gamma: {} '.format(g), end =" ")
        fair_model = GerryFairClassifier(C=100, printflag=False, gamma=g, max_iters=iters)
        fair_model.gamma=g
        fair_model.fit(dataset)
        predictions = array_to_tuple((fair_model.predict(dataset)).labels)
        _, fp_diff = fp_auditor.audit(predictions)
        _, fn_diff = fn_auditor.audit(predictions)
        fp_violations.append(fp_diff)
        fn_violations.append(fn_diff)

    plt.plot(fp_violations, fn_violations, label='adult')
    plt.xlabel('False Positive Disparity')
    plt.ylabel('False Negative Disparity')
    plt.legend()
    plt.title('FP vs FN Unfairness')
    plt.savefig('gerryfair_fp_fn.png')
    plt.close()

gamma_list = [0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.05]
fp_vs_fn(data_set, gamma_list, pareto_iters)
Image(filename='gerryfair_fp_fn.png')
