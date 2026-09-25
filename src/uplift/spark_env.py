"""Windows-local PySpark environment defaults.

PySpark on Windows needs JAVA_HOME, a winutils.exe/hadoop.dll on PATH via HADOOP_HOME
(otherwise Parquet writes fail with "HADOOP_HOME and hadoop.home.dir are unset"), and
matching PYSPARK_PYTHON/PYSPARK_DRIVER_PYTHON (otherwise the Python worker can fail to
connect back). These are only applied as fallback defaults (`setdefault`) when the
environment doesn't already set them, so a properly configured machine (or CI on Linux,
where none of this applies) is untouched.
"""

import os
import sys

_DEFAULT_WINDOWS_JAVA_HOME = r"C:\Users\trilo\AppData\Local\dev-tools\jdk-17"
_DEFAULT_WINDOWS_HADOOP_HOME = r"C:\Users\trilo\AppData\Local\dev-tools\hadoop-3.3.6"


def apply_windows_spark_defaults() -> None:
    if os.name != "nt":
        return
    os.environ.setdefault("JAVA_HOME", _DEFAULT_WINDOWS_JAVA_HOME)
    os.environ.setdefault("HADOOP_HOME", _DEFAULT_WINDOWS_HADOOP_HOME)
    hadoop_bin = os.path.join(os.environ["HADOOP_HOME"], "bin")
    if hadoop_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


LOCAL_SPARK_CONFIG = {
    "spark.driver.host": "127.0.0.1",
    "spark.driver.bindAddress": "127.0.0.1",
}
