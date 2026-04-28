import unittest


class HealthCheckTests(unittest.TestCase):
    def test_health_report_shape(self) -> None:
        from tools.health_check import gather_health_report

        report = gather_health_report()
        self.assertIn("status", report)
        self.assertIn("checks", report)
        self.assertIsInstance(report["checks"], list)

    def test_health_check_targets_supported_web_stack(self) -> None:
        from tools.health_check import gather_health_report

        report = gather_health_report()
        checks = {item["name"]: item for item in report["checks"]}
        self.assertIn("repo_layout", checks)
        self.assertIn("core_modules", checks)
        self.assertIn("api_runtime_config", checks)
        self.assertNotIn("streamlit_app/main.py", checks["repo_layout"]["detail"])
        self.assertNotIn("streamlit", checks["core_modules"]["detail"])
        self.assertNotIn("azure_openai_env", checks)


if __name__ == "__main__":
    unittest.main()
