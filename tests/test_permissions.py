"""Hardware-free tests for kuiskaus.permissions (issue #64).

The Cocoa/Quartz calls are patched at the module boundary so no real TCC
state is read or changed.
"""

from unittest.mock import MagicMock, patch

import pytest

from kuiskaus import permissions


def _patch_status(mic: int = 3, listen: bool = True, ax: bool = True):
    av = MagicMock()
    av.AVMediaTypeAudio = "soun"
    av.AVCaptureDevice.authorizationStatusForMediaType_.return_value = mic
    quartz = MagicMock()
    quartz.CGPreflightListenEventAccess.return_value = listen
    return (
        patch.object(permissions, "AVFoundation", av),
        patch.object(permissions, "Quartz", quartz),
        patch.object(permissions, "AXIsProcessTrusted", return_value=ax),
    )


def _missing(**kwargs) -> list[str]:
    p_av, p_q, p_ax = _patch_status(**kwargs)
    with p_av, p_q, p_ax:
        return permissions.missing_permissions()


def test_all_granted_reports_nothing_missing():
    assert _missing() == []


@pytest.mark.parametrize("mic", [0, 1, 2])
def test_microphone_not_authorized_is_missing(mic):
    assert _missing(mic=mic) == ["Microphone"]


def test_input_monitoring_and_accessibility_reported_in_stable_order():
    assert _missing(mic=2, listen=False, ax=False) == [
        "Microphone",
        "Input Monitoring",
        "Accessibility",
    ]


def test_open_settings_uses_the_pane_deep_link():
    with patch.object(permissions.subprocess, "run") as run:
        permissions.open_settings("Input Monitoring")
    run.assert_called_once_with(
        [
            "open",
            (
                "x-apple.systempreferences:com.apple.preference.security"
                "?Privacy_ListenEvent"
            ),
        ],
        check=False,
    )


def test_open_settings_unknown_name_is_a_noop():
    with patch.object(permissions.subprocess, "run") as run:
        permissions.open_settings("Bogus")
    run.assert_not_called()


def test_request_microphone_prompts_only_when_undetermined():
    p_av, p_q, p_ax = _patch_status(mic=0)
    with p_av as av, p_q, p_ax:
        permissions.request_microphone()
    av.AVCaptureDevice.requestAccessForMediaType_completionHandler_.assert_called_once()

    p_av, p_q, p_ax = _patch_status(mic=2)
    with p_av as av, p_q, p_ax:
        permissions.request_microphone()
    av.AVCaptureDevice.requestAccessForMediaType_completionHandler_.assert_not_called()
