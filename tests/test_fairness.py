import numpy as np
import pandas as pd
from student_dropout.fairness_summary import summarize


def test_undefined_group_rate_does_not_imply_parity():
    data = pd.DataFrame({
        'Attribute': ['Age', 'Age'], 'Outcome': ['Dropout', 'Dropout'],
        'Count': [20, 30], 'Selection rate': [0.2, 0.6],
        'TPR': [np.nan, 0.8], 'FPR': [0.1, 0.3],
        'Small group': [True, False], 'Sparse outcome support': [True, False],
    })
    result = summarize(data).iloc[0]
    assert np.isclose(result['Demographic parity difference'], 0.4)
    assert np.isclose(result['Selection min/max ratio'], 1/3)
    assert np.isnan(result['Equalised odds difference'])
