"""Cross-layer relational feature tests."""
import pandas as pd
from defense.cross_layer import cross_layer_features


def test_cross_layer_summaries_are_pairwise_and_finite():
    rows = []
    for layer, scale in [("conv1", 1.0), ("conv2", 2.0)]:
        for feature, value in [("mean", 1.0), ("std", 0.5), ("l1", 5.0), ("l2", 3.0),
                               ("max_abs", 2.0), ("sparsity", 0.0), ("skewness", 0.0), ("kurtosis", 0.0)]:
            rows.append({"experiment_id": "e", "round": 1, "client_id": 1,
                         "layer_name": layer, "feature_name": feature, "feature_value": value * scale})
    result = cross_layer_features(pd.DataFrame(rows))
    assert len(result) == 1
    assert 0.0 <= result.iloc[0].summary_cosine <= 1.0
    assert result.iloc[0].relative_l2_magnitude > 0
