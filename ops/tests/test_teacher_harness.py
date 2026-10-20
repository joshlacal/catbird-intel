"""Re-export teacher harness acceptance tests under ops/tests for discoverability."""

from teacher_harness.tests.test_teacher_harness import TeacherHarnessAcceptanceTests

__all__ = ["TeacherHarnessAcceptanceTests"]
