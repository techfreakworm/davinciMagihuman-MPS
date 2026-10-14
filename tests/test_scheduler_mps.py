import unittest
from unittest import mock

import torch

from inference.pipeline.scheduler_unipc import FlowUniPCMultistepScheduler


@unittest.skipUnless(torch.backends.mps.is_available(), "needs an MPS device")
class SchedulerOnMPSMatchesCPU(unittest.TestCase):
    def schedulers(self):
        cpu, mps = FlowUniPCMultistepScheduler(), FlowUniPCMultistepScheduler()
        cpu.set_timesteps(8, device="cpu", shift=5)
        mps.set_timesteps(8, device="mps", shift=5)
        return cpu, mps

    def test_eight_unipc_steps(self):
        for shape in [(1, 48, 4, 4, 4), (1, 17, 64)]:
            with self.subTest(shape=shape):
                torch.manual_seed(0)
                cpu, mps = self.schedulers()
                x = torch.randn(shape)
                y = x.to("mps")
                for i in range(8):
                    v = torch.randn(shape)
                    x = cpu.step(v, cpu.timesteps[i], x, return_dict=False)[0]
                    y = mps.step(v.to("mps"), mps.timesteps[i], y, return_dict=False)[0]
                    torch.testing.assert_close(y.cpu(), x, rtol=2e-5, atol=2e-5)

    def test_ddim_and_sde_steps(self):
        cpu, mps = self.schedulers()
        x = torch.ones(1, 2, 3)
        v = torch.full_like(x, 0.5)

        def fixed_noise(shape, **kwargs):
            return torch.full(shape, 0.25, device=kwargs["device"], dtype=kwargs["dtype"])

        with mock.patch("inference.pipeline.scheduler_unipc.randn_tensor", side_effect=fixed_noise):
            for method in ("step_ddim", "step_sde"):
                with self.subTest(method=method):
                    expected = getattr(cpu, method)(v, 3, x)
                    actual = getattr(mps, method)(v.to("mps"), 3, x.to("mps"))
                    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
