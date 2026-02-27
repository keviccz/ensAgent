import unittest


class HealthCheckTests(unittest.TestCase):
    def test_health_report_shape(self) -> None:
        from tools.health_check import gather_health_report

        report = gather_health_report()
        self.assertIn("status", report)
        self.assertIn("checks", report)
        self.assertIsInstance(report["checks"], list)


if __name__ == "__main__":
    unittest.main()
