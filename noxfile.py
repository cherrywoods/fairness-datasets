import nox


@nox.session
@nox.parametrize(
    'python,torch_version,numpy_version,pandas_version,requests_version',
    [
        ("3.8", "1.8.0", "1.18.0", "1.0.0", "2.23.0"),
        ("3.10", "1.12.1", "1.22.0", "1.4.0", "2.27.0"),
        ("3.11", "2.0.1", "1.25.2", "2.0.3", "2.31.0"),
    ],
    ids=["python3.8", "python3.10", "python3.11"]
)
def tests(session, torch_version, numpy_version, pandas_version, requests_version):
    session.install(f"torch=={torch_version}")
    session.install(f"numpy=={numpy_version}")
    session.install(f"pandas=={pandas_version}")
    session.install(f"requests=={requests_version}")
    session.install('pytest')
    session.install('.')
    session.run('pytest')
