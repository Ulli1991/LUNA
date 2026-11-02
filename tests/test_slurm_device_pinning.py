"""Which GPUs a process under SLURM claims.

One task per GPU (the scaling runs) pins each process to its localid. One
task per node (a process owning both GPUs of a node, the layout a multi-node
Torch worker needs) must take all of them: pinning it to localid 0 left the
second GPU idle and the mesh short by half.
"""
import pytest

from luna.setup.jax_init import _local_device_ids_from_slurm

SLURM = ("SLURM_LOCALID", "SLURM_NTASKS", "SLURM_NNODES", "SLURM_STEP_NUM_TASKS",
         "SLURM_STEP_NUM_NODES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
         "HIP_VISIBLE_DEVICES")


@pytest.fixture
def env(monkeypatch):
    for k in SLURM:
        monkeypatch.delenv(k, raising=False)

    def set_(**kv):
        for k, v in kv.items():
            monkeypatch.setenv(k, str(v))
    return set_


def test_one_task_per_gpu_pins_to_the_localid(env):
    env(SLURM_LOCALID=1, SLURM_STEP_NUM_TASKS=4, SLURM_STEP_NUM_NODES=2)
    assert _local_device_ids_from_slurm() == [1]


def test_one_task_per_node_takes_every_gpu(env):
    """Explicitly: given None, JAX's SLURM detection pins to the localid."""
    env(SLURM_LOCALID=0, SLURM_STEP_NUM_TASKS=2, SLURM_STEP_NUM_NODES=2,
        ROCR_VISIBLE_DEVICES="0,1")
    assert _local_device_ids_from_slurm() == [0, 1]


def test_one_task_per_node_counts_gpus_without_a_visible_list(env):
    env(SLURM_LOCALID=0, SLURM_STEP_NUM_TASKS=2, SLURM_STEP_NUM_NODES=2,
        SLURM_GPUS_ON_NODE=2)
    assert _local_device_ids_from_slurm() == [0, 1]


def test_job_level_counts_when_the_step_ones_are_absent(env):
    env(SLURM_LOCALID=0, SLURM_NTASKS=2, SLURM_NNODES=2, ROCR_VISIBLE_DEVICES="0,1")
    assert _local_device_ids_from_slurm() == [0, 1]
    env(SLURM_NTASKS=4)
    assert _local_device_ids_from_slurm() == [0]


def test_a_per_task_binding_to_one_gpu_is_index_zero(env):
    env(SLURM_LOCALID=1, SLURM_STEP_NUM_TASKS=4, SLURM_STEP_NUM_NODES=2,
        ROCR_VISIBLE_DEVICES=1)
    assert _local_device_ids_from_slurm() == [0]


def test_a_single_task_is_left_to_jax(env):
    env(SLURM_LOCALID=0, SLURM_STEP_NUM_TASKS=1, SLURM_STEP_NUM_NODES=1)
    assert _local_device_ids_from_slurm() is None
