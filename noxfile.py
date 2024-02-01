import nox


@nox.session
@nox.parametrize(
    "python,torch_version,numpy_version,pandas_version,requests_version",
    [
        ("3.9", "1.9.0", "1.20.0", "2.1.0", "2.25.0"),
        ("3.10", "1.12.1", "1.22.0", "2.1.0", "2.27.0"),
        ("3.11", "2.1.2", "1.26.2", "2.1.4", "2.31.0"),
        ("3.12", "2.2.0", "1.26.3", "2.2.0", "2.31.0"),
    ],
    ids=["python3.9", "python3.10", "python3.11", "python3.12"],
)
def tests(session, torch_version, numpy_version, pandas_version, requests_version):
    session.install(f"torch=={torch_version}")
    session.install(f"numpy=={numpy_version}")
    session.install(f"pandas=={pandas_version}")
    session.install(f"requests=={requests_version}")
    session.install("pytest")
    session.install(".")
    session.run("pytest")
