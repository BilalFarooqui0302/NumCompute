import unittest
import numpy as np
from numcompute_stream import (DecisionTreeClassifier, StandardScaler, Imputer,
    OneHotEncoder, StreamingStats, StreamingClassificationMetrics, Pipeline, StreamTrainer)
from numcompute_stream.visualise import plot_metric_over_time, compare_models, plot_predictions_vs_ground_truth

class TestDecisionTree(unittest.TestCase):
    def setUp(self):
        self.X=np.array([[0],[1],[2],[3],[4],[5]],float); self.y=np.array([0,0,0,1,1,1])
    def test_perfect_fit(self): self.assertEqual(DecisionTreeClassifier().fit(self.X,self.y).score(self.X,self.y),1)
    def test_gini(self): self.assertEqual(DecisionTreeClassifier(criterion='gini').fit(self.X,self.y).score(self.X,self.y),1)
    def test_entropy(self): self.assertEqual(DecisionTreeClassifier(criterion='entropy').fit(self.X,self.y).score(self.X,self.y),1)
    def test_depth_zero_majority(self): self.assertTrue(np.all(DecisionTreeClassifier(max_depth=0).fit(self.X,self.y).predict(self.X)==0))
    def test_tie_smallest_label(self): self.assertEqual(DecisionTreeClassifier(max_depth=0).fit([[0],[1]],[1,0]).predict([[2]])[0],0)
    def test_buffer_waits(self):
        m=DecisionTreeClassifier(buffer_size=10); m.partial_fit(self.X[:3],self.y[:3]); self.assertEqual(m.retrain_count_,0); self.assertEqual(m.buffered_samples_,3)
    def test_buffer_retrains(self):
        m=DecisionTreeClassifier(buffer_size=6); m.partial_fit(self.X[:3],self.y[:3]); m.partial_fit(self.X[3:],self.y[3:]); self.assertEqual(m.retrain_count_,1); self.assertEqual(m.buffered_samples_,0)
    def test_predict_before_fit(self):
        with self.assertRaises(RuntimeError): DecisionTreeClassifier().predict(self.X)
    def test_bad_criterion(self):
        with self.assertRaises(ValueError): DecisionTreeClassifier(criterion='bad')
    def test_bad_depth(self):
        with self.assertRaises(ValueError): DecisionTreeClassifier(max_depth=-1)
    def test_bad_buffer(self):
        with self.assertRaises(ValueError): DecisionTreeClassifier(buffer_size=0)
    def test_bad_shapes(self):
        with self.assertRaises(ValueError): DecisionTreeClassifier().fit([[1],[2]],[1])
    def test_feature_mismatch(self):
        m=DecisionTreeClassifier().fit([[1],[2]],[0,1])
        with self.assertRaises(ValueError): m.predict([[1,2]])
    def test_nan_features(self):
        X=np.array([[0],[1],[np.nan],[4],[5],[np.nan]])
        m=DecisionTreeClassifier().fit(X,[0,0,0,1,1,1]); self.assertEqual(len(m.predict(X)),6)
    def test_constant_feature_leaf(self):
        m=DecisionTreeClassifier().fit(np.ones((4,1)),[0,1,1,1]); self.assertTrue(np.all(m.predict(np.ones((2,1)))==1))
    def test_string_labels(self):
        m=DecisionTreeClassifier().fit([[0],[1]],np.array(['a','b'])); self.assertEqual(m.predict([[0]])[0],'a')
    def test_max_features(self):
        m=DecisionTreeClassifier(max_features=1,random_state=3).fit(np.c_[self.X,self.X],self.y); self.assertEqual(len(m.predict(np.c_[self.X,self.X])),6)
    def test_chunk_overflow_trains_on_entire_buffer(self):
        m=DecisionTreeClassifier(buffer_size=3); m.partial_fit(self.X[:4],self.y[:4]); self.assertEqual(m.retrain_count_,1); self.assertEqual(m.buffered_samples_,0)
    def test_nan_route_is_deterministic(self):
        X=np.array([[0.],[1.],[4.],[5.],[np.nan]])
        m=DecisionTreeClassifier().fit(X,[0,0,1,1,0]); self.assertEqual(m.predict([[np.nan]])[0],0)
    def test_predict_proba_sums_to_one(self):
        m=DecisionTreeClassifier().fit(self.X,self.y); np.testing.assert_allclose(m.predict_proba(self.X).sum(axis=1),1)
    def test_reset_stream(self):
        m=DecisionTreeClassifier(buffer_size=2).partial_fit(self.X[:2],self.y[:2]); m.reset_stream(); self.assertEqual(m.buffered_samples_,0)

class TestPreprocessing(unittest.TestCase):
    def test_scaler_partial_fit_mean(self):
        s=StandardScaler().partial_fit([[1,2]]).partial_fit([[3,4]]); np.testing.assert_allclose(s.mean_,[2,3])
    def test_scaler_variance(self):
        s=StandardScaler().fit([[1],[3]]); np.testing.assert_allclose(s.var_,[1])
    def test_scaler_nan(self):
        s=StandardScaler().fit([[1],[np.nan],[3]]); np.testing.assert_allclose(s.mean_,[2])
    def test_scaler_constant(self):
        s=StandardScaler().fit([[2],[2]]); np.testing.assert_allclose(s.transform([[2]]),[[0]])
    def test_scaler_unfitted(self):
        with self.assertRaises(RuntimeError): StandardScaler().transform([[1]])
    def test_scaler_feature_change(self):
        with self.assertRaises(ValueError): StandardScaler().partial_fit([[1]]).partial_fit([[1,2]])
    def test_imputer_mean(self):
        i=Imputer().partial_fit([[1,np.nan],[3,4]]); np.testing.assert_allclose(i.transform([[np.nan,2]]),[[2,2]])
    def test_imputer_all_missing(self):
        i=Imputer().fit([[np.nan],[np.nan]]); self.assertEqual(i.transform([[np.nan]])[0,0],0)
    def test_imputer_constant(self):
        i=Imputer(strategy='constant',fill_value=-1).fit([[np.nan]]); self.assertEqual(i.transform([[np.nan]])[0,0],-1)
    def test_encoder_growth(self):
        e=OneHotEncoder().partial_fit([['a']]); self.assertEqual(e.n_features_out_,1); e.partial_fit([['b']]); self.assertEqual(e.n_features_out_,2)
    def test_encoder_unknown_ignore(self):
        e=OneHotEncoder().fit([['a']]); np.testing.assert_array_equal(e.transform([['z']]),[[0]])
    def test_encoder_unknown_error(self):
        e=OneHotEncoder(handle_unknown='error').fit([['a']])
        with self.assertRaises(ValueError): e.transform([['z']])

class TestStatsMetrics(unittest.TestCase):
    def test_stats_chunk_mean(self):
        s=StreamingStats().update_stats([[1,2]]).update_stats([[3,4]]); np.testing.assert_allclose(s.result()['mean'],[2,3])
    def test_stats_variance(self):
        s=StreamingStats().update_stats([[1],[3]]); np.testing.assert_allclose(s.variance_,[1])
    def test_stats_nan(self):
        s=StreamingStats().update_stats([[1],[np.nan],[3]]); np.testing.assert_allclose(s.result()['count'],[2])
    def test_stats_quantile(self):
        s=StreamingStats().update_stats([[1],[3],[2]]); np.testing.assert_allclose(s.quantile(.5),[2])
    def test_stats_histogram(self):
        h,c=StreamingStats().update_stats([[1],[2],[3]]).histogram(bins=2); self.assertEqual(h.sum(),3)
    def test_stats_bad_quantile(self):
        with self.assertRaises(ValueError): StreamingStats().update_stats([[1]]).quantile(2)
    def test_metric_accuracy(self):
        m=StreamingClassificationMetrics().update([0,1],[0,0]); self.assertEqual(m.result()['accuracy'],.5)
    def test_metric_confusion(self):
        m=StreamingClassificationMetrics().update([0,1],[0,1]); self.assertEqual(int(m.result()['confusion_matrix'].trace()),2)
    def test_metric_reset(self):
        m=StreamingClassificationMetrics().update([0],[0]); m.reset(); self.assertEqual(len(m._y),0)
    def test_metric_rolling(self):
        m=StreamingClassificationMetrics(window_size=2).update([0,1,1],[0,0,1]); self.assertEqual(m.result(rolling=True)['accuracy'],.5)
    def test_metric_auc(self):
        m=StreamingClassificationMetrics().update([0,1,0,1],[0,1,0,1],[.1,.9,.2,.8]); self.assertEqual(m.result()['roc_auc'],1)
    def test_metric_auc_ties(self):
        m=StreamingClassificationMetrics().update([0,1],[0,1],[0.5,0.5]); self.assertEqual(m.result()['roc_auc'],0.5)
    def test_metric_auc_incomplete_scores(self):
        m=StreamingClassificationMetrics().update([0,1],[0,1]).update([0,1],[0,1],[.1,.9]); self.assertTrue(np.isnan(m.result()['roc_auc']))
    def test_stats_reset(self):
        s=StreamingStats().update_stats([[1]]); s.reset(); self.assertIsNone(s.count_)
    def test_metric_shape_error(self):
        with self.assertRaises(ValueError): StreamingClassificationMetrics().update([1,2],[1])

class TestIntegrationVisual(unittest.TestCase):
    def test_pipeline(self):
        p=Pipeline([('imp',Imputer()),('tree',DecisionTreeClassifier(buffer_size=4))]); p.partial_fit([[0],[1]],[0,0]); p.partial_fit([[3],[4]],[1,1]); self.assertEqual(len(p.predict([[0],[4]])),2)
    def test_pipeline_encoder_schema_growth_resets_model(self):
        tree=DecisionTreeClassifier(buffer_size=2)
        p=Pipeline([('encode',OneHotEncoder()),('tree',tree)])
        p.partial_fit(np.array([['a'],['a']],dtype=object),[0,0])
        self.assertEqual(tree.retrain_count_,1)
        p.partial_fit(np.array([['b'],['b']],dtype=object),[1,1])
        self.assertEqual(tree.retrain_count_,1)
        np.testing.assert_array_equal(tree.classes_,[1])
        self.assertEqual(tree.buffered_samples_,0)
    def test_stream_trainer_logs(self):
        m=DecisionTreeClassifier(buffer_size=2); t=StreamTrainer(m); log=t.fit_chunk([[0],[1]],[0,1]); self.assertEqual(log['chunk'],1)
    def test_visual_metric(self):
        fig=plot_metric_over_time([.5,.7],show=False); self.assertIsNotNone(fig)
    def test_visual_compare(self):
        fig=compare_models([.5,.6],[.4,.8],show=False); self.assertIsNotNone(fig)
    def test_visual_predictions(self):
        fig=plot_predictions_vs_ground_truth([0,1],[0,0],show=False); self.assertIsNotNone(fig)
    def test_visual_shape_error(self):
        with self.assertRaises(ValueError): plot_predictions_vs_ground_truth([1],[1,2],show=False)

if __name__=='__main__': unittest.main()
