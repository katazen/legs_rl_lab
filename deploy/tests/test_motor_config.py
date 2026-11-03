"""Hardware-free checks of shared motor profiles and effort frame handling."""
from pathlib import Path
import sys
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motor_config import FB_SIGN, load_motors, detect_effort_frame


def test_profiles_ids_and_legacy_direction_are_preserved():
    cfg = load_motors()
    assert [j['model'] for j in cfg['joints']].count('small') == 2
    assert [j['name'] for j in cfg['joints'] if j['model'] == 'small'] == ['L6', 'R6']
    assert cfg['models']['large'] == cfg['models']['small'] == {
        'position_range': 12.5, 'velocity_range': 45., 'effort_range': 30.}
    np.testing.assert_array_equal(FB_SIGN, [1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1])


def test_new_effort_is_not_double_flipped_and_old_effort_is_converted():
    pd = np.sin(np.arange(300)[:, None] * .1 + np.arange(12)[None]) * 3
    new, _ = detect_effort_frame(pd, pd)
    old, _ = detect_effort_frame(pd, pd * FB_SIGN)
    np.testing.assert_array_equal(new, 1)
    np.testing.assert_array_equal(old, FB_SIGN)
    with pytest.raises(ValueError, match='无法判定'):
        detect_effort_frame(pd, np.zeros_like(pd))


@pytest.mark.parametrize('bad', ['direction', 'model', 'range', 'order', 'id'])
def test_invalid_hardware_configuration_fails_before_use(tmp_path, bad):
    cfg = load_motors()
    if bad == 'direction': cfg['joints'][0]['direction'] = 0
    if bad == 'model': cfg['joints'][5]['model'] = 'large'
    if bad == 'range': cfg['models']['small']['effort_range'] = float('nan')
    if bad == 'order': cfg['joints'].reverse()
    if bad == 'id': cfg['joints'][4]['slave_id'] = 5
    file = tmp_path / 'motors.yaml'
    file.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError): load_motors(file)
    import subprocess
    driver = Path(__file__).resolve().parents[1] / 'control_ws/install/armcontrol/lib/armcontrol/arm_control_node'
    result = subprocess.run([str(driver), '--ros-args', '-p', 'preflight_only:=true',
                             '-p', f'motor_config_file:={file}'], capture_output=True, text=True, timeout=5)
    assert result.returncode == 1
    assert 'Motor configuration PASS' not in result.stdout + result.stderr
    assert 'Self-developed dual-leg motors enabled' not in result.stdout + result.stderr
