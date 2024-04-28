import numpy as np
import python_speech_features.sigproc as psf_sigproc
import torch

from kws.libs.speech_features import sigproc


def test_magspec():
    np.random.seed(0)
    signal = np.random.rand(16000)
    frame_len = 400  # 25ms
    frame_step = 160  # 10ms

    frames_np = psf_sigproc.framesig(signal, frame_len, frame_step)
    frame_torch = sigproc.framesig(torch.tensor(signal), frame_len, frame_step)

    NFFT = 512
    magspec_np = psf_sigproc.magspec(frames_np, NFFT)
    magspec_torch = sigproc.magspec(frame_torch, NFFT).numpy()

    np.testing.assert_allclose(magspec_np, magspec_torch, rtol=1e-5, atol=1e-8)
