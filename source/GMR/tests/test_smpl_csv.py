import csv

import numpy as np
import pytest

from general_motion_retargeting.utils.smpl_csv import (
    CSV_COLUMNS,
    as_gmr_smplx_data,
    load_gvhmr_pt,
    normalize_smpl_params,
    read_smpl_csv,
    write_smpl_csv,
)


def sample(frames=4, pose_3d=False, beta_shape="vector"):
    pose = np.arange(frames * 63, dtype=np.float32).reshape(frames, 63) / 100
    if pose_3d:
        pose = pose.reshape(frames, 21, 3)
    beta = np.arange(10, dtype=np.float32)
    if beta_shape == "one":
        beta = beta[None]
    elif beta_shape == "frames":
        beta = np.broadcast_to(beta, (frames, 10)).copy()
    return {
        "global_orient": np.arange(frames * 3).reshape(frames, 3) / 10,
        "body_pose": pose,
        "transl": np.arange(frames * 3).reshape(frames, 3) / 5,
        "betas": beta,
    }


def test_csv_schema_frames_and_round_trip(tmp_path):
    source = sample(beta_shape="frames")
    path = tmp_path / "motion.csv"
    write_smpl_csv(path, source, 50)
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == CSV_COLUMNS
    assert len(rows[0]) == 80
    assert len(rows) == 5
    restored, fps = read_smpl_csv(path)
    assert fps == 50
    for key in source:
        np.testing.assert_allclose(restored[key], source[key], rtol=1e-6)
    gmr = as_gmr_smplx_data(restored, fps)
    np.testing.assert_allclose(gmr["root_orient"], source["global_orient"])
    np.testing.assert_allclose(gmr["pose_body"], source["body_pose"])
    np.testing.assert_allclose(gmr["trans"], source["transl"])
    np.testing.assert_allclose(gmr["betas"], source["betas"])
    assert gmr["mocap_frame_rate"].item() == 50


def test_vector_betas_are_broadcast():
    normalized = normalize_smpl_params(sample(frames=3))
    assert normalized["betas"].shape == (3, 10)
    np.testing.assert_array_equal(normalized["betas"][0], normalized["betas"][2])


def test_joint_axis_angle_body_pose_is_flattened():
    normalized = normalize_smpl_params(sample(frames=2, pose_3d=True))
    assert normalized["body_pose"].shape == (2, 63)


def test_pt_uses_global_params_and_tensor_inputs(tmp_path):
    torch = pytest.importorskip("torch")
    source = {key: torch.as_tensor(value) for key, value in sample().items()}
    path = tmp_path / "hmr4d_results.pt"
    torch.save({"smpl_params_global": source, "smpl_params_incam": {}, "fps": 50}, path)
    restored, fps = load_gvhmr_pt(path)
    assert fps == 50
    np.testing.assert_allclose(restored["transl"], sample()["transl"])


@pytest.mark.parametrize("bad_key,bad_value", [
    ("global_orient", np.zeros((4, 4))),
    ("body_pose", np.zeros((4, 22, 3))),
    ("transl", np.zeros((3, 3))),
    ("betas", np.zeros((4, 11))),
])
def test_invalid_shapes_raise_clear_errors(bad_key, bad_value):
    data = sample()
    data[bad_key] = bad_value
    with pytest.raises(ValueError, match=bad_key if bad_key != "transl" else "frame count mismatch"):
        normalize_smpl_params(data)


def test_missing_column_and_inconsistent_fps(tmp_path):
    path = tmp_path / "motion.csv"
    write_smpl_csv(path, sample(), 50)
    rows = list(csv.reader(path.open()))
    rows[0].pop()
    for row in rows[1:]:
        row.pop()
    broken = tmp_path / "missing.csv"
    with broken.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    with pytest.raises(ValueError, match="missing columns"):
        read_smpl_csv(broken)

    write_smpl_csv(path, sample(), 50)
    rows = list(csv.reader(path.open()))
    rows[2][0] = "25"
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    with pytest.raises(ValueError, match="inconsistent"):
        read_smpl_csv(path)


def test_explicit_fps_conflict_in_pt(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "motion.pt"
    torch.save({"smpl_params_global": sample(), "fps": 30}, path)
    with pytest.raises(ValueError, match="conflicts"):
        load_gvhmr_pt(path, fps=50)
