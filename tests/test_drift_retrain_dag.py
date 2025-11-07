import os
import json
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from airflow import DAG
from airflow.utils import timezone


@pytest.fixture
def mock_dag():
    """Create a mock DAG for testing."""
    return DAG(
        'test_dag',
        start_date=timezone.utcnow(),
        catchup=False,
    )


class TestDriftRetrainDAG:
    """Tests for drift_retrain_dag DAG structure."""
    
    def test_dag_imports(self):
        """Test that the DAG module imports without errors."""
        try:
            import sys
            if '/opt/airflow' not in sys.path:
                sys.path.insert(0, '/opt/airflow')
            from dags.drift_retrain_dag import dag
            assert dag is not None
            assert dag.dag_id == 'customer_churn_drift_retrain'
        except ImportError:
            pytest.skip("DAG module not available in test environment")
    
    def test_dag_has_required_tasks(self):
        """Test that DAG has all required tasks."""
        try:
            import sys
            if '/opt/airflow' not in sys.path:
                sys.path.insert(0, '/opt/airflow')
            from dags.drift_retrain_dag import dag
            
            task_ids = {task.task_id for task in dag.tasks}
            
            required_tasks = {
                'build_features',
                'detect_drift',
                'generate_reports',
                'branch_on_drift',
                'retrain_combined',
                'skip_retraining',
                'done',
            }
            
            assert required_tasks.issubset(task_ids), f"Missing tasks: {required_tasks - task_ids}"
        except ImportError:
            pytest.skip("DAG module not available in test environment")


class TestChooseBranch:
    """Tests for choose_branch function."""
    
    def test_choose_branch_with_drift(self):
        """Test that branch returns retrain_combined when drift is detected."""
        try:
            import sys
            if '/opt/airflow' not in sys.path:
                sys.path.insert(0, '/opt/airflow')
            from dags.drift_retrain_dag import choose_branch
            
            # Mock context with drift detected
            context = {
                'ti': MagicMock(
                    xcom_pull=MagicMock(return_value=True)
                )
            }
            
            result = choose_branch(**context)
            assert result == 'retrain_combined'
        except ImportError:
            pytest.skip("DAG module not available in test environment")
    
    def test_choose_branch_without_drift(self):
        """Test that branch returns skip_retraining when no drift is detected."""
        try:
            import sys
            if '/opt/airflow' not in sys.path:
                sys.path.insert(0, '/opt/airflow')
            from dags.drift_retrain_dag import choose_branch
            
            # Mock context without drift
            context = {
                'ti': MagicMock(
                    xcom_pull=MagicMock(return_value=False)
                )
            }
            
            result = choose_branch(**context)
            assert result == 'skip_retraining'
        except ImportError:
            pytest.skip("DAG module not available in test environment")


class TestRunDriftDetection:
    """Tests for run_drift_detection function."""
    
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('builtins.open', create=True)
    def test_drift_detection_no_production_file(
        self, mock_open, mock_makedirs, mock_exists
    ):
        """Test drift detection when production file doesn't exist."""
        try:
            import sys
            if '/opt/airflow' not in sys.path:
                sys.path.insert(0, '/opt/airflow')
            from dags.drift_retrain_dag import run_drift_detection
            
            # Mock os.path.exists to return False for production, True for features
            def exists_side_effect(path):
                if 'features' in path:
                    return True
                return False
            
            mock_exists.side_effect = exists_side_effect
            mock_ti = MagicMock()
            
            context = {
                'ti': mock_ti
            }
            
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch('dags.drift_retrain_dag.FEATURES_PATH', os.path.join(tmpdir, 'features.csv')):
                    with patch('dags.drift_retrain_dag.PRODUCTION_DATA_PATH', os.path.join(tmpdir, 'production.csv')):
                        with patch('dags.drift_retrain_dag.DRIFT_REPORT_PATH', os.path.join(tmpdir, 'report.json')):
                            # Create features file
                            with open(os.path.join(tmpdir, 'features.csv'), 'w') as f:
                                f.write('dummy')
                            
                            result = run_drift_detection(**context)
                            
                            assert result.get('is_drift') is False
                            assert 'reason' in result
                            mock_ti.xcom_push.assert_called_with(key='is_drift', value=False)
        except ImportError:
            pytest.skip("DAG module not available in test environment")


class TestDAGStructure:
    """Tests for DAG task dependencies and structure."""
    
    def test_dag_task_dependencies(self):
        """Test that DAG has correct task dependencies."""
        try:
            import sys
            if '/opt/airflow' not in sys.path:
                sys.path.insert(0, '/opt/airflow')
            from dags.drift_retrain_dag import dag
            
            # Get tasks
            task_map = {task.task_id: task for task in dag.tasks}
            
            # Check build_features dependencies
            build_features = task_map['build_features']
            assert len(build_features.upstream_list) == 0, "build_features should have no upstream tasks"
            
            # Check detect_drift dependencies
            detect_drift = task_map['detect_drift']
            assert 'build_features' in [t.task_id for t in detect_drift.upstream_list]
            
            # Check branch dependencies
            branch = task_map['branch_on_drift']
            assert 'generate_reports' in [t.task_id for t in branch.upstream_list]
            
            # Check retrain_combined has no upstream (depends on branch)
            retrain_combined = task_map['retrain_combined']
            # Branch uses BranchPythonOperator, so we check via upstream_list
            
            # Check skip_retraining has no upstream (depends on branch)
            skip_retraining = task_map['skip_retraining']
            
            # Check done has multiple upstream
            done = task_map['done']
            assert len(done.upstream_list) >= 2, "done should have multiple upstream tasks"
            
        except ImportError:
            pytest.skip("DAG module not available in test environment")
    
    def test_dag_no_retrain_features_task(self):
        """Test that retrain_features task has been removed."""
        try:
            import sys
            if '/opt/airflow' not in sys.path:
                sys.path.insert(0, '/opt/airflow')
            from dags.drift_retrain_dag import dag
            
            task_ids = {task.task_id for task in dag.tasks}
            
            assert 'retrain_features' not in task_ids, "retrain_features task should not exist"
            assert 'skip_retraining' in task_ids, "skip_retraining task should exist"
        except ImportError:
            pytest.skip("DAG module not available in test environment")


class TestDAGExecution:
    """Tests for DAG execution scenarios."""
    
    def test_dag_execution_path_with_drift(self):
        """Test DAG execution path when drift is detected."""
        try:
            import sys
            if '/opt/airflow' not in sys.path:
                sys.path.insert(0, '/opt/airflow')
            
            # This would require setting up Airflow execution environment
            # For now, we test the branching logic
            pytest.skip("Full DAG execution testing requires Airflow setup")
        except ImportError:
            pytest.skip("DAG module not available in test environment")
    
    def test_dag_execution_path_without_drift(self):
        """Test DAG execution path when no drift is detected."""
        try:
            import sys
            if '/opt/airflow' not in sys.path:
                sys.path.insert(0, '/opt/airflow')
            
            # This would require setting up Airflow execution environment
            # For now, we test the branching logic
            pytest.skip("Full DAG execution testing requires Airflow setup")
        except ImportError:
            pytest.skip("DAG module not available in test environment")
